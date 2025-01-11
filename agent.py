from datetime import datetime
import logging
import os

import aiohttp
from dotenv import load_dotenv
from pymongo.errors import PyMongoError
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
from livekit.plugins import openai, deepgram, silero
import pymongo


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

class InterviewQuestionsFnc(llm.FunctionContext):
    @llm.ai_callable()
    def get_interview_questions(
        self,
        domain: Annotated[
            str, llm.TypeInfo(description="The most comfortable professional domain of the interviewee")
        ]
    ):
        """
        Retrieves a list of interview questions tailored to the interviewee's most comfortable domain.
        
        Provides targeted questions to highlight the candidate's strengths and expertise.
        """
        # Predefined interview questions dictionary
        interview_questions = {
            "software_engineering": [
                "Describe the most complex software system you've designed.",
                "What's your preferred methodology for software development?",
                "How do you stay updated with new technologies?"
            ], 
            "data_science": [
                "Walk me through your typical data analysis workflow.",
                "How do you handle missing or inconsistent data?",
                "Describe a machine learning project that had a significant impact.",
                "What metrics do you consider when evaluating model performance?",
                "How do you communicate complex technical findings to non-technical stakeholders?"
            ],
            "product_management": [
                "How do you prioritize features for a product roadmap?",
                "Describe a product you improved based on user feedback.",
                "How do you balance user needs with business objectives?",
                "Walk me through your product development process.",
                "How do you measure the success of a product?"
            ],
            "machine_learning": [
                "Explain a challenging machine learning problem you've solved.",
                "How do you approach feature engineering?",
                "Describe your experience with different ML algorithms.",
                "How do you handle overfitting and underfitting?",
                "What's your approach to model interpretability?"
            ],
            "cybersecurity": [
                "Describe a security vulnerability you've identified or mitigated.",
                "How do you stay current with emerging security threats?",
                "Walk me through your incident response strategy.",
                "How do you balance security with user experience?",
                "Describe your approach to threat modeling."
            ],
            "general": [
                "Tell me about yourself and your professional journey.",
                "What are your greatest professional strengths?",
                "Describe a significant challenge you've overcome at work.",
                "Where do you see your career progressing?",
                "What motivates you professionally?"
            ]
        }
        
        # Normalize input and handle unknown domains
        domain = domain.lower().replace(" ", "_")
        if domain not in interview_questions:
            domain = "general"
        
        # Return the list of questions
        return interview_questions[domain]

    @llm.ai_callable()
    def save_single_interview_answer(
        self,
        user_id: Annotated[
            str, llm.TypeInfo(description="Unique identifier for the user taking the interview")
        ],
        question: Annotated[
            str, llm.TypeInfo(description="Current interview question")
        ],
        answer: Annotated[
            str, llm.TypeInfo(description="User's answer to the current question")
        ],
        interview_domain: Annotated[
            str, llm.TypeInfo(description="Professional domain of the interview")
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
            db = client["interview-app"]
            answers_collection = db["answers"]
            
            # Prepare single answer document
            answer_document = {
                "user_id": user_id,
                "domain": interview_domain,
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
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are lexi, a voice assistant that conduct interviews. Ask the questions one by one in the order provided. Save each question along with its given by the user in exact words"
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # This project is configured to use Deepgram STT, OpenAI LLM and TTS plugins
    # Other great providers exist like Cartesia and ElevenLabs
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins
    assistant = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(),   
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )

    assistant.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await assistant.say("Hey There, I am lexi and I am here to conduct your interview, shall we begin ?")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
