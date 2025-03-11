import asyncio
import os
from datetime import datetime
import logging
import shutil
from pathlib import Path
from string import punctuation
import time
import uuid

import typer
import json

import yaml


app = typer.Typer()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s: %(message)s",
)
logging.getLogger("asyncio").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


async def command_worker(
    pending_queue: asyncio.Queue,
    running_queue: list,
    queue_semaphore: asyncio.Semaphore,
    id: int,
    running_file: Path,
    complete_file: Path,
    output_dir: Path = None,
    gpu_id: int = None,
):
    # TODO: Currently a hacky workaround to a lack of queue peek is using the semaphore
    #   ideally we would have a peek method on the queue
    output_dir = output_dir or Path.cwd()
    translation = str.maketrans("", "", punctuation)

    logging.info(f"Starting command worker {id} - piping to {output_dir}")
    while True:
        logging.info(f"Command worker {id} waiting for command")
        try:
            # timeout = 5*60  
            # command = await asyncio.wait_for(pending_queue.get(), timeout=timeout)
            command = await pending_queue.get()
        except asyncio.TimeoutError:
            logging.warning(f"Command worker {id} timed out waiting for command, shutting down.")
            break


        if command is None:
            break
        
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logging.info(f"Command worker {id} assigned to GPU {gpu_id}")

        random_id = str(uuid.uuid4())
        command = f"{command.strip()} # UUID: {random_id}"
        # FIXME: Need to get command ID working for cancel and command worker; so we can repeat identical commands
        logging.debug(f"Command worker {id} running {random_id} command: {command}")
        with open(running_file, "a") as f:
            f.write(command)

        files = {}
        for output_dest in ["stdout", "stderr"]:
            command_path = command.translate(translation).strip().replace(" ", "_")
            save_path = output_dir / f"{command_path}/{output_dest}"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            files[output_dest] = open(save_path, "w")
            logging.info(f"Command worker {id} saving {output_dest} to {save_path}")
        
        command_parts = command.strip().split(" ")
        logging.info(f"Command worker {id} running command: {command_parts}")
        # process = await asyncio.create_subprocess_exec(command_parts[0], *command_parts[1:], env=env, **files)
        # FIXME: Update to use exec after UUID working
        process = await asyncio.create_subprocess_shell(command, env=env, **files)
        start = time.time()
        task = asyncio.create_task(process.wait(), name=command)
        running_queue.append(task)
        try:
            await task
            logging.info(f"Command worker {id} finished command: {command}")
            running_queue.remove(task)
            with open(running_file, "r") as f:
                commands = f.readlines()
            with open(running_file, "w") as f:
                other_commands = [x for x in commands if x != command]
                f.writelines(other_commands)
            with open(complete_file, "a") as f:
                completed_in = time.time() - start
                if completed_in > 3600:
                    completed_in = f"{completed_in/3600:.2f}h"
                elif completed_in > 60:
                    completed_in = f"{completed_in/60:.2f}m"
                else:
                    completed_in = f"{completed_in:.2f}s"
                completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{completed_at} ({completed_in}): {command.strip()}\n")
            for file in files.values():
                file.close()
        except asyncio.CancelledError:
            process.terminate()
            await process.wait()
            logging.info(f"Command worker {id} cancelled command: {command}")
        except KeyboardInterrupt as e:
            process.kill()
            await process.wait()
            logging.warning(f"Keyboard interrupt, cancelling command: {command}")
            raise e

        queue_semaphore.release()
        pending_queue.task_done()


async def cancel_worker(running_queue: list[asyncio.Task], running_file: Path, polling_interval=1):
    logging.info("Starting cancel worker")
    while True:
        tasks_to_cancel = []
        with open(running_file, "r") as f:
            commands = f.readlines()
        for task in running_queue:
            if task.get_name() not in commands:
                tasks_to_cancel.append(task)
        if not tasks_to_cancel:
            await asyncio.sleep(polling_interval)
            continue

        logging.info(f"Found discrepancy in running tasks and command file.")
        logging.info(f"Cancelling tasks: {[task.get_name() for task in tasks_to_cancel]}")
        for task in tasks_to_cancel:
            task.cancel()
            running_queue.remove(task)
            with open(running_file, "w") as f:
                f.writelines([f"{x.get_name()}\n" for x in running_queue])
        logging.debug(f"Cancelled {len(tasks_to_cancel)} tasks")
        await asyncio.sleep(polling_interval)


async def queue_worker(
    pending_queue: asyncio.Queue,
    can_add_to_pending: asyncio.Semaphore,
    queue_file: Path,
    polling_interval=1,
):
    logging.info("Starting queue worker")
    while True:
        logging.debug("Queue worker waiting for space on runners")
        await can_add_to_pending.acquire()
        logging.debug("Queue worker has space on runners, looking for command")
        with open(queue_file, "r") as f:
            commands = f.readlines()
        if commands:
            command = commands[0]
            logging.debug(
                f"Queue worker found command, attempting to put on pending queue: {command}"
            )
            await pending_queue.put(command)
            logging.info(f"Queue worker put {command} in pending queue")
            with open(queue_file, "w") as f:
                f.writelines(commands[1:])
            logging.debug(f"Queue worker removed {command} from queue file")
        else:
            can_add_to_pending.release()
        await asyncio.sleep(polling_interval)


async def main(num_workers: int, queue_file: Path, running_file: Path, out_dir: str, gpu_assignments: dict):
    out_dir = Path(out_dir)
    pending_queue = asyncio.Queue(maxsize=num_workers)
    can_add_to_pending = asyncio.BoundedSemaphore(num_workers)
    running_queue = []
    completed_file = running_file.with_name(f"{running_file.stem}_completed.txt")
    with open(running_file, "w") as f:
        f.write("")
    with open(completed_file, "a") as f:
        f.write("")
    workers = []
    for i in range(num_workers):
        workers.append(
            asyncio.create_task(
                command_worker(
                    pending_queue,
                    running_queue,
                    can_add_to_pending,
                    id=i,
                    output_dir=out_dir,
                    running_file=running_file,
                    complete_file=completed_file,
                    gpu_id=gpu_assignments.get(i, None),
                )
            )
        )
    workers.append(asyncio.create_task(cancel_worker(running_queue, running_file=running_file)))
    workers.append(
        asyncio.create_task(queue_worker(pending_queue, can_add_to_pending, queue_file=queue_file))
    )
    try:
        results = await asyncio.gather(*workers, return_exceptions=False)
    except KeyboardInterrupt as e:
        # FIXME: This is currently not working; should really just raise a cancel on the running file
        logging.warning("Keyboard interrupt, cancelling other running tasks")
        for task in running_queue:
            task.cancel()
        raise e


@app.command()
def entrypoint(
    worker_config_file: Path,
    gpu_config_file: Path = None,
):
    """
    Run the command queue processor with optional GPU assignments.

    Worker config file should contain:
    - number of workers
    - output_dir
    - running_filename
    - queue_filename

    This script processes a queue of commands using multiple workers. Each worker can optionally be assigned to a specific GPU.

    GPU assignments can be specified in two ways:
    1. Using the --gpu-assignments option with a JSON string.
    2. Using the --gpu-config-file option with a path to a JSON file.

    If both options are provided, --gpu-assignments takes precedence.
    If neither option is provided, no GPU assignments will be made.

    Example usage:
    1. With GPU config file:
       python runner_asyncio.py --worker-config-file default_config.yaml --gpu-config-file gpu_config.json

    """
    with open(worker_config_file, "r") as f:
        worker_config = yaml.safe_load(f)
    num_workers = worker_config["num_workers"]
    out_dir = Path(worker_config["stdout_dir"])
    running_file = Path(worker_config["running_file"])
    queue_file = Path(worker_config["queue_file"])
    

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_out_dir = typer.confirm(f"Clear out_dir {out_dir} before running?")
    if clear_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    
    
    if not queue_file.exists():
        queue_file.parent.mkdir(parents=True, exist_ok=True)
        queue_file.touch()
    if not running_file.exists():
        running_file.parent.mkdir(parents=True, exist_ok=True)
        running_file.touch()
    

    # Parse GPU assignments
    gpu_assign_dict = {}

    try:
        with open(gpu_config_file, 'r') as f:
            gpu_assign_dict = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        typer.echo(f"Error reading GPU config file: {gpu_config_file}")
        raise typer.Exit(code=1)

    # Convert worker IDs to integers
    gpu_assign_dict = {int(k): v for k, v in gpu_assign_dict.items()}

    asyncio.run(
        main(
            num_workers=num_workers,
            out_dir=out_dir,
            queue_file=queue_file,
            running_file=running_file,
            gpu_assignments=gpu_assign_dict,
        )
    )

if __name__ == "__main__":
    app()