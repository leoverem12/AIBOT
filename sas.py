import os
import discord
import logging
from discord import app_commands
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio
import datetime
import aiohttp
import re
import tempfile
import chardet
import time
from collections import defaultdict
import threading
import moviepy as mp
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-exp")

MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_INPUT_LENGTH = 2000
MAX_OUTPUT_LENGTH = 2000
MAX_CONTEXT_LENGTH = 3000
MAX_LOG_FILE_SIZE = 100 * 1024 * 1024
LOG_FILE = "conversation_history.txt"
ERROR_LOG_FILE = "error_log.txt"
MAX_RPM = 15
REQUEST_COUNTS = defaultdict(int)
LAST_REQUEST_TIME = defaultdict(float)
RATE_LIMIT_ACTIVE = False
MAX_VIDEO_DURATION = 15


def log_message(sender, message, user_id=None, is_error=False):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = ERROR_LOG_FILE if is_error else LOG_FILE
    try:
      with open(log_file, "a", encoding="utf-8") as f:
             f.write(f"{timestamp} - {sender}: {message}\n")
    except Exception as e:
        logger.error(f"Logging error: {e}", exc_info=True)


def check_log_size():
    try:
      for log_file in [LOG_FILE, ERROR_LOG_FILE]:
        if os.path.exists(log_file) and os.path.getsize(log_file) > MAX_LOG_FILE_SIZE:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(log_file, "w", encoding="utf-8") as f:
                f.writelines(lines[-50:])
    except Exception as e:
        logger.error(f"Log check error: {e}", exc_info=True)

def is_russian(text):
    return not any(char in text.lower() for char in "єіїґ")

async def download_image(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 429:
                    return None, "Too many requests. Try later."
                if response.status == 404:
                    return None, "Image not found."
                if response.status != 200:
                    return None, f"Error downloading image: HTTP {response.status}"
                return await response.read(), None
    except Exception as e:
        logger.error(f"Download image error from {url}: {e}", exc_info=True)
        return None, "Failed to download image."


async def process_image(image_data, image_url, prompt):
  try:
    if image_data:
        full_prompt = f"Image: \n {image_url or ''} \n {prompt}\n Recognize text (Ukrainian), describe structure and colors."
        parts = [full_prompt, {"mime_type": "image/png", "data": image_data}]
        response = await model.generate_content_async(parts)
        return response, None
    return None, "Image analysis not supported."
  except Exception as e:
      logger.error(f"Image process error: {e}", exc_info=True)
      return None, "Failed to process image." if "An internal error has occurred" not in str(e) else "Google server error. Try again."


async def process_video(video_data, video_url, prompt):
    tmp_video_path = None
    tmp_audio_path = None
    tmp_image_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(video_data)
            tmp_video_path = tmp_file.name

        clip = mp.VideoFileClip(tmp_video_path)
        if clip.duration > MAX_VIDEO_DURATION:
            clip.close()
            os.remove(tmp_video_path)
            return None, f"Video duration exceeds the limit ({MAX_VIDEO_DURATION} seconds)."


        audio_buffer = None
        if clip.audio:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio_file:
                tmp_audio_path = tmp_audio_file.name
                clip.audio.write_audiofile(tmp_audio_path, codec='mp3')
                with open(tmp_audio_path, "rb") as f:
                    audio_buffer = f.read()
        
        frame = clip.get_frame(0.0)
        clip.close()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_image_file:
              tmp_image_path = tmp_image_file.name
              mp.ImageClip(frame).save_frame(tmp_image_path, t=0.0, with_mask=False)

              with open(tmp_image_path, "rb") as f:
                 image_buffer = io.BytesIO(f.read())


        parts = []
        if audio_buffer:
              parts.append({"mime_type": "audio/mp3", "data": audio_buffer})
        parts.append({"mime_type": "image/png", "data": image_buffer.read()})
        
        full_prompt = f"Video: \n {video_url or ''} \n {prompt}\n Analyze the content, focus on text, actions, scene and objects. Use Ukrainian language."
        parts.insert(0, full_prompt)
        response = await model.generate_content_async(parts)
        return response, None

    except Exception as e:
        logger.error(f"Video process error: {e}", exc_info=True)
        return None, "Failed to process video. Ensure that video is not corrupted and length less then 15 seconds."
    finally:
        if tmp_video_path and os.path.exists(tmp_video_path):
             os.remove(tmp_video_path)
        if tmp_audio_path and os.path.exists(tmp_audio_path):
              os.remove(tmp_audio_path)
        if tmp_image_path and os.path.exists(tmp_image_path):
              os.remove(tmp_image_path)


async def process_file(file_data, prompt):
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file_data)
            tmp_file_path = tmp_file.name
        with open(tmp_file_path, "rb") as f:
            encoding = chardet.detect(f.read())['encoding'] or 'utf-8'
            try:
              file_content = f.read().decode(encoding)
            except UnicodeDecodeError:
                file_content = f.read().decode("utf-8", errors='replace')
                logger.warning(f"File read with unknown encoding. Replaced invalid characters.")
        full_prompt = f"{prompt} \n Analyze text, find mistakes, describe structure.\n {file_content}"
        response = await model.generate_content_async(full_prompt)
        return response, None
    except Exception as e:
        logger.error(f"File process error: {e}", exc_info=True)
        return None, "Failed to process file. It might be non-text or corrupted."
    finally:
         if tmp_file_path:
              os.remove(tmp_file_path)

async def generate_response(user_name, prompt, message, image_data=None, image_url=None, file_data=None, video_data=None, video_url=None):
    for attempt in range(MAX_RETRIES):
        try:
             user_id = message.author.id
             full_prompt = f"Нове питання від {user_name}: {prompt}\n"

             if image_data or image_url:
                image_response, error = await process_image(image_data, image_url, full_prompt)
                if error:
                     await message.reply(error)
                     return None
                response = image_response
             elif video_data or video_url:
                video_response, error = await process_video(video_data, video_url, full_prompt)
                if error:
                     await message.reply(error)
                     return None
                response = video_response
             elif file_data:
                file_response, error = await process_file(file_data, full_prompt)
                if error:
                     await message.reply(error)
                     return None
                response = file_response
             else:
                full_prompt = f"{full_prompt} \n Згенеруй нейтральну та об'єктивну відповідь українською мовою, використовуючи не більше 150 слів."
                response = await model.generate_content_async(full_prompt)

             if response and response.text:
                return response.text[:MAX_OUTPUT_LENGTH]
             else:
                 error_msg = f"Порожня відповідь від Gemini, finish_reason: {getattr(response, 'finish_reason', 'N/A')}, safety_ratings: {getattr(response, 'safety_ratings', 'N/A')}"
                 logger.warning(error_msg)
                 log_message("Bot", error_msg, user_id, is_error=True)

                 if getattr(response, 'finish_reason', None) == 3:
                     await message.reply("Не вдалося згенерувати відповідь через обмеження безпеки. Спробуйте перефразувати.")
                     return None
                 elif response and "An internal error has occurred" in str(response):
                     error_msg = "Помилка сервера Google. Контекст може бути занадто довгим. Спробуйте скоротити свій запит або зачекайте."
                     log_message("Bot", f"Помилка 500: {error_msg}", user_id, is_error=True)
                     await message.reply(error_msg)
                     return None
                 else:
                    await message.reply("Не вдалося згенерувати відповідь.")
                    return None
        except Exception as e:
                logger.error(f"Помилка генерації (спроба {attempt + 1}): {e}", exc_info=True)
                log_message("Bot", f"Помилка генерації: {e}, спроба {attempt+1}", user_id, is_error=True)
                await asyncio.sleep(RETRY_DELAY)
    await message.reply("Не вдалося згенерувати відповідь. Спробуйте пізніше.")
    return None

async def check_rate_limit(message):
    global RATE_LIMIT_ACTIVE
    user_id = message.author.id
    current_time = time.time()
    
    if user_id not in LAST_REQUEST_TIME or current_time - LAST_REQUEST_TIME[user_id] >= 60:
        REQUEST_COUNTS[user_id] = 0
        LAST_REQUEST_TIME[user_id] = current_time
    
    REQUEST_COUNTS[user_id] += 1
    
    if REQUEST_COUNTS[user_id] > MAX_RPM:
      if not RATE_LIMIT_ACTIVE:
        RATE_LIMIT_ACTIVE = True
        await message.reply(f"You've exceeded the rate limit ({MAX_RPM} requests per minute). Wait a minute.")
        await asyncio.sleep(60)
        REQUEST_COUNTS[user_id] = 0
        LAST_REQUEST_TIME[user_id] = time.time()
        RATE_LIMIT_ACTIVE = False
      else:
        await message.reply("Rate limit is active. Try again in a few seconds.")
      return True
    return False

@client.event
async def on_ready():
    await tree.sync()
    logger.info(f"Bot {client.user} activated!")
    await client.change_presence(activity=discord.Activity(name="Ask me anything!", type=discord.ActivityType.playing))

@client.event
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.mention_everyone:
      return
    if await check_rate_limit(message):
      return
    
    image_url, image_data, file_data, video_url, video_data = None, None, None, None, None
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image"):
                image_data = await attachment.read()
                image_url = attachment.url
                break
            elif attachment.content_type and attachment.content_type.startswith("video"):
                video_data = await attachment.read()
                video_url = attachment.url
                break
            else:
                file_data = await attachment.read()

    if client.user.mentioned_in(message):
        try:
            query = message.content.replace(f'<@{client.user.id}>', '').strip()
            if not (image_data or image_url) and any(word in query.lower() for word in ["зображення", "картинку"]):
              await message.reply("Please attach an image.")
              return
            if not (video_data or video_url) and any(word in query.lower() for word in ["відео", "ролик", "запис"]):
              await message.reply("Please attach a video.")
              return
            if len(query) > MAX_INPUT_LENGTH:
                await message.reply(f"Request too long. Max length: {MAX_INPUT_LENGTH} symbols.")
                return
            user_name, user_id = message.author.name, message.author.id
            if image_data or image_url:
              await message.reply("Analyzing image...")
            if file_data:
              await message.reply("Analyzing file...")
            if video_data or video_url:
              await message.reply("Analyzing video...")
            
            async with message.channel.typing():
                response = await generate_response(user_name, query, message, image_data, image_url, file_data, video_data, video_url)
                if not response:
                   return

                if len(response) > MAX_OUTPUT_LENGTH:
                    await message.reply(f"Response too long. Max length: {MAX_OUTPUT_LENGTH} symbols.")
                    log_message("Bot", f"Response was cut due to length ({MAX_OUTPUT_LENGTH}).", user_id, is_error=True)
                    return


                elif "файл" in query.lower() and not file_data:
                     await message.reply("Please provide a file.")
                elif file_data and "файл" in query.lower():
                     await message.reply(f"Here's what I found!\n {response}")
                elif "як мене звати" in query.lower():
                    await message.reply(f"Glad to help!\n {response}")
                else:
                   await message.reply(f"{response}")
            log_message(message.author.name, query, user_id)
            if not response.startswith("Failed to generate response"):
                log_message("Bot", response, user_id)
            check_log_size()
            logger.info(f"Request from {message.author.name}: {query}")
            logger.info(f"Bot response: {response}")

        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            log_message("Bot", f"Error: {e}", user_id, is_error=True)
            await message.reply("An error occurred processing the request.")

async def send_console_message(channel_id, message_text):
    try:
        channel = client.get_channel(int(channel_id))
        if channel:
            await channel.send(message_text)
            log_message("Console", message_text)
            logger.info(f"Console message sent to {channel_id}: {message_text}")
        else:
            logger.error(f"Channel {channel_id} not found.")
    except Exception as e:
        logger.error(f"Console send error: {e}", exc_info=True)

def console_input_handler():
    while True:
        try:
            user_input = input("Enter channel ID and message ('123456789 Hello'): ")
            if not user_input: continue
            parts = user_input.split(" ", 1)
            if len(parts) < 2:
              print("Invalid format. Enter channel ID and message.")
              continue
            channel_id, message_text = parts
            asyncio.run_coroutine_threadsafe(send_console_message(channel_id, message_text), client.loop)
        except Exception as e:
            logger.error(f"Console input error: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        console_thread = threading.Thread(target=console_input_handler, daemon=True)
        console_thread.start()
        client.run(os.getenv("DISCORD_TOKEN"))
    except Exception as e:
        logger.critical(f"Startup error: {e}", exc_info=True)