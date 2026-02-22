import uuid
import importlib.util
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

AGENT_DIR = Path(__file__).parent / "QA-Agent"
AGENT_MODULE_PATH = AGENT_DIR / "agent.py"

if not AGENT_MODULE_PATH.exists():
    raise FileNotFoundError(f"Missing agent module at {AGENT_MODULE_PATH}")

spec = importlib.util.spec_from_file_location("qa_agent_module", AGENT_MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load spec for {AGENT_MODULE_PATH}")

qa_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa_agent_module)
qa_agent = qa_agent_module.qa_agent

load_dotenv()

session_service = InMemorySessionService()

initialstate = {
    "username": "Brandon Hancock",
    "preferences": """
 I love cars and driving.
    My favorite food is Indian food.
    My favorite TV show is Fargo.
    Loves it when people like and subscribe to his YouTube channel.
    """,
}

#creating a new session
APP_NAME = "session Agent"
USER_ID = "sxv"
SESSION_ID = str(uuid.uuid4())

session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state=initialstate,
)

print("CREATED A NEW SESSION:")
print(f"\tSession ID: {SESSION_ID}")

runner = Runner(
    agent=qa_agent,
    app_name="QA Agent",
    session_service=session_service,
)
new_message = types.Content(
    role="user", parts=[types.Part(text="What is Brandon's favorite TV show?")]
)

for event in runner.run(
    user_id=USER_ID,
    session_id=SESSION_ID,
    new_message=new_message,
):
    if event.is_final_response():
        if event.content and event.content.parts:
            print(f"Final Response: {event.content.parts[0].text}")

print("==== Session Event Exploration ====")
session = session_service.get_session(
    app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
)

# Log final Session state
print("=== Final Session State ===")
for key, value in session.state.items():
    print(f"{key}: {value}")