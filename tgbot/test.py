import asyncio
import logging
import sys
from os import getenv
import fileinput


from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, MenuButtonType
from aiogram.filters import CommandStart
from aiogram.types import Message, KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder

# Bot token can be obtained via https://t.me/BotFather
TOKEN = "7486950870:AAG_m1QUZOZHz0-gs56ov7Yh5OAOCmJIyik"

# All handlers should be attached to the Router (or Dispatcher)

dp = Dispatcher()



def log(user_name, text):
    with open('log.txt', 'a') as f:
        f.write(f'user_name: {user_name}, text:{text}\n')


is_log = True



def replace_in_file(file_path, user_id, new_text):
    is_done = False
    with fileinput.input(file_path, inplace=True) as file:
        for line in file:
            line_array = line.split(', ')
            if line_array[0] ==  user_id:
                new_line = line.replace(line, new_text)
                print(new_line, end='')
                is_done = True
            else:
                print(line, end='')
    return is_done


def change_user_prefernces(user_id, lang):
    if not replace_in_file('preferences.txt', str(user_id), f'{user_id}, {lang}\n'):
        with open('preferences.txt', 'a') as f:
            f.write(f'{user_id}, {lang}\n')



def get_user_preferences(user_id):
    with open('preferences.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            if line[0] == str(user_id):
                return line
        
    change_user_prefernces(user_id, lang = 'ru')
    return None


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    kb = [
        [KeyboardButton(text="Язык"), KeyboardButton(text="Помощь")]
    ]
    keyboard = ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True,)
    if is_log:
        log(message.from_user.full_name, message.text)
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!", reply_markup=keyboard)





@dp.message(F.text.lower() == "язык")
async def language(message: Message):
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(
        text="Ru",
        callback_data="lang_ru")
    )
    builder.add(InlineKeyboardButton(
        text="En",
        callback_data="lang_en")
    )
    preferences = get_user_preferences(message.from_user.id)
    if preferences:
        reply = f"У вас выбран язык: {preferences[1]}, но вы можете сменить его в любой момент"
    else:
        reply = "У вас еще не выбран язык, по умолчанию установлен русский"

    await message.answer(
        reply,
        reply_markup=builder.as_markup()
    )


@dp.callback_query(F.data == 'lang_ru')
async def change_user_language(callback: CallbackQuery):
    lang = F.data.split('_')[1]
    change_user_prefernces(callback.from_user.id, lang)
    await callback.message.answer(f'Вы успешно изменили язык на: {lang}')
    await callback.answer()


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())