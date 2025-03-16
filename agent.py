from datetime import datetime
import logging
import os

import aiohttp
from dotenv import load_dotenv
from pymongo.errors import PyMongoError
from bson.objectid import ObjectId
from typing import Annotated, List
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, silero, openai
import pymongo


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


class InterviewQuestionsFnc(llm.FunctionContext):
    @llm.ai_callable()
    async def get_interview_questions(
        self,
        interview_id: Annotated[
            str, llm.TypeInfo(description="Unique identifier for the interview")
        ],
    ):
        """
        Retrieves a list of interview questions for the interview question set based on the interview's question_set_id.
        """
        try:
            # Connect to MongoDB
            client = pymongo.MongoClient(os.environ.get("MONGODB_URI"))
            db = client["cafi_db"]
            interviews_collection = db["interviews"]

            # Fetch the interview document using the provided interview_id
            interview = interviews_collection.find_one({"_id": ObjectId(interview_id)})
            print("fetched interview", interview)
            if not interview:
                raise ValueError(f"Interview with id {interview_id} not found.")

            question_set_id = interview["question_set_id"]
            question_set_id_str = str(question_set_id)
            questions_collection = db["questions"]

            # Retrieve all questions with the matching question_set_id
            interview_question_set = questions_collection.find({"question_set_id": question_set_id_str})
            # Return the questions as a newline-separated string
            questions = [question["text"] for question in interview_question_set]
            return "\n".join(questions)

        except PyMongoError as e:
            # Log the error
            print(f"Error fetching interview questions: {e}")
            raise

    @llm.ai_callable()
    async def save_single_interview_answer(
        self,
        user_id: Annotated[
            str, llm.TypeInfo(description="Unique identifier for the user taking the interview")
        ],
        interview_id: Annotated[
            str, llm.TypeInfo(description="Unique identifier for the interview")
        ],
        question: Annotated[
            str, llm.TypeInfo(description="Current interview question")
        ],
        answer: Annotated[
            str, llm.TypeInfo(description="User's answer to the current question")
        ],
        question_number: Annotated[
            int, llm.TypeInfo(description="Current question number in the interview sequence")
        ]
    ):
        """
        Saves a single interview question and answer to the MongoDB database.
        
        Allows incremental saving of interview responses during the interview process.
        """
        try:
            # Connect to MongoDB
            client = pymongo.MongoClient(os.environ.get("MONGODB_URI"))
            db = client["cafi_db"]
            answers_collection = db["answers"]

            # Prepare single answer document
            answer_document = {
                "user_id": user_id,
                "interview_id": interview_id,
                "timestamp": datetime.now(),
                "question_number": question_number,
                "question": question,
                "answer": answer
            }

            # Insert the document
            result = answers_collection.insert_one(answer_document)

            # Close the connection
            client.close()

            # Return the inserted document's ID
            return str(result.inserted_id)

        except PyMongoError as e:
            # Log the error 
            print(f"Error saving interview answer: {e}")
            raise


fnc_ctx = InterviewQuestionsFnc()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    logger.info("Waiting for participant...")
    participant = await ctx.wait_for_participant()

    # Extract user_id and interview_id from participant identity (assuming the format "userId_interviewId")
    identity = participant.identity
    user_id, interview_id = identity.split("_", 1)
    logger.info(f"Starting voice assistant for {user_id} (interview {interview_id})")

    # Improved prompt for the agent
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            f"You are Lexi, an AI interviewer conducting interview {interview_id} for user {user_id}. "
            "Follow this exact workflow:\n"
            "1. INTRODUCTION PHASE:\n"
            "   - Fetch the questions for the interview using the provided interview_id.\n"
            "   - Do not engage in casual conversation.\n\n"
            "2. QUESTION HANDLING PHASE:\n"
            "   - Ask questions exactly as provided in the list, one at a time and in strict order.\n"
            "   - After each answer, immediately save the exact question/answer pair and move to the next question.\n\n"
            "3. CONCLUSION PHASE:\n"
            "   - After the final answer is saved, state 'Thank you, that concludes our interview'.\n"
            "   - Briefly acknowledge the completion and shut down the session.\n\n"
            "RULES:\n"
            "- Never deviate from the provided question list.\n"
            "- No follow-up questions.\n"
            "- If the user asks unrelated questions, respond with: 'Let's focus on the current interview question please'.\n"
            "- End session after the final answer is stored and acknowledge the completion of the interview."
        ),
    )

    assistant = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=deepgram.tts.TTS(model="aura-asteria-en"),
        chat_ctx=initial_ctx,
        max_nested_fnc_calls=5,
        fnc_ctx=fnc_ctx,
    )

    assistant.start(ctx.room, participant)

    await assistant.say("Hey there, I am Lexi and I am here to conduct your interview. Shall we begin?")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
