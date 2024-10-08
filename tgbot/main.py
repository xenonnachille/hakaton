import asyncio
import logging
import sys
from os import getenv
import fileinput
from language import LANG
POLS = {'language':['язык', 'language'],
        'help':['помощь', 'help']
        }
from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, MenuButtonType
from aiogram.filters import CommandStart
from aiogram.types import Message, KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder


TOKEN = "7486950870:AAG_m1QUZOZHz0-gs56ov7Yh5OAOCmJIyik"


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
    Этот обработчик реагирует на команду   `/start`
    """
    prefs = get_user_preferences(message.from_user.id)
    if prefs:
        lang = prefs[1].strip()
    else:
        lang = 'ru'
    kb = [
        [KeyboardButton(text=LANG[lang]['language']), KeyboardButton(text=LANG[lang]['help'])]
    ]
    keyboard = ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True,)
    if is_log:
        log(message.from_user.full_name, message.text)
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!", reply_markup=keyboard)





@dp.message(lambda x: x.text.lower() in POLS['language'])
async def language(message: Message):
    builder = InlineKeyboardBuilder()
    builder.add(InlineKeyboardButton(
        text="🇷🇺Ru",
        callback_data="lang_ru")
    )
    builder.add(InlineKeyboardButton(
        text="🇬🇧En",
        callback_data="lang_en")
    )
    preferences = get_user_preferences(message.from_user.id)
    if preferences:
        lang = preferences[1].strip()
        reply = f"{LANG[lang]['choose_language'][0]}{LANG[lang]['language_name']}{LANG[lang]['choose_language'][1]}"
    else:
        reply = "У вас еще не выбран язык, по умолчанию установлен русский"

    await message.answer(
        reply,
        reply_markup=builder.as_markup()
    )


@dp.message(lambda x: x.text.lower() in POLS['help'])
async def language(message: Message):
    prefs = get_user_preferences(message.from_user.id)
    if prefs:
        lang = prefs[1].strip()
    else:
        lang = 'ru'
    await message.answer(
        LANG[lang]['help_text']
    )


@dp.callback_query(F.data.startswith('lang'))
async def change_user_language(callback: CallbackQuery):
    lang = callback.data.split('_')[1]
    print(lang)
    change_user_prefernces(callback.from_user.id, lang)
    kb = [
        [KeyboardButton(text=LANG[lang]['language']), KeyboardButton(text=LANG[lang]['help'])]
    ]
    keyboard = ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True,)
    await callback.message.answer(f"{LANG[lang]['choose_success']} {LANG[lang]['language_name']}", reply_markup=keyboard)
    
    await callback.answer()



@dp.message()
async def language(message: Message):
    log(message.from_user.full_name, message.text)

async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())