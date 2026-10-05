import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Phone data -> Panda -> SmolVLA")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Validate provenance and retarget an annotated video"
    )
    prepare.add_argument("manifest")
    prepare.add_argument("--output", required=True)
    tasks = commands.add_parser("tasks", help="List LIBERO task IDs and instructions")
    tasks.add_argument("--suite", default="libero_goal")
    rollout = commands.add_parser("rollout", help="Execute phone-derived targets or a trained VLA")
    rollout.add_argument("trajectory")
    rollout.add_argument("--output", required=True)
    rollout.add_argument("--suite", default="libero_goal")
    rollout.add_argument("--task-id", type=int, default=0)
    rollout.add_argument("--init-id", type=int, default=0)
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("--resolution", type=int, default=256)
    rollout.add_argument("--policy", help="Local fine-tuned SmolVLA pretrained_model directory")
    rollout.add_argument("--device", default="cpu")
    export = commands.add_parser("export", help="Export successful train-split rollouts to LeRobot")
    export.add_argument("runs", nargs="+")
    export.add_argument("--repo-id", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare

        result = prepare(args.manifest, args.output)
    elif args.command == "tasks":
        from .simulation import list_tasks

        result = list_tasks(args.suite)
    elif args.command == "rollout":
        from .simulation import rollout

        result = rollout(args)
    else:
        from .lerobot_bridge import export

        result = export(args)
    print(json.dumps(result, indent=2))
