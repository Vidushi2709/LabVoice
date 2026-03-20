import logging

from livekit.agents import JobContext, RoomInputOptions, WorkerOptions, cli
from livekit.plugins import noise_cancellation

from assistant import Assistant
from eval import EvalTracker
from session import build_session

logger = logging.getLogger(__name__)


# entrypoint
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    eval_tracker = EvalTracker()
    session = build_session()

    # Attach all metrics (existing EOU/usage + new barge-in/repair/judge)
    eval_tracker.attach(session, ctx)

    await session.start(
        agent=Assistant(eval_tracker=eval_tracker),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))