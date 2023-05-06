import glob
import os
import shutil
import requests
import torch.cuda
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
import subprocess
from tqdm.contrib.telegram import tqdm, trange
from random import random, randint
from time import sleep
import telebot
from telebot import types
import warnings
import numpy as np

warnings.filterwarnings("ignore")

token = 'YOUR_TOKEN_HERE'
bot = telebot.TeleBot(token)
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

os.makedirs("photo", exist_ok=True)
os.makedirs("sticker", exist_ok=True)
os.makedirs("sticker_pack", exist_ok=True)
os.makedirs("result_photo", exist_ok=True)

USER_MAP = {}
USER = {
    "save": 1,
    "sticker_set_name": "",
    "img_name": ""
}


def parce_sticker_set(message, sticker_set_name):
    print(f"PARCE SET FOR USER {message.from_user.id}...")

    try:
        sticker_set = bot.get_sticker_set(sticker_set_name)
        stickers = sticker_set.stickers
        for idx, stick in enumerate(stickers):
            if stick.is_video or stick.is_animated:
                bot.send_message(message.from_user.id, "Отправьте неанимированные стикеры.")
                return
            file_id = stick.file_id
            result = bot.get_file(file_id)
            response = requests.get(f"https://api.telegram.org/file/bot{token}/{result.file_path}")
            img = response.content

            f = open(f"sticker_pack/{message.from_user.id}/{idx}.webp", "wb")
            f.write(img)
            f.close()
        bot.send_message(message.from_user.id, "Пожалуйста, отправьте Ваше фото")
    except Exception as e:
        print("54", e)
        bot.send_message(message.from_user.id, "Данного стикерпака не существует")


def tune_dreambooth(message):
    print(f"TUNE DREAMBOOTH FOR USER {message.from_user.id}...")

    try:
        instance_data_dir = f'--instance_data_dir=sticker_pack/{message.from_user.id}'
        outp_dir = f'--output_dir=tuned_model_sticker_pack/{message.from_user.id}'

        script = subprocess.run(
            ['accelerate', 'launch', 'train_model.py', '--pretrained_model_name_or_path=tuned_model256_lowlr',
             instance_data_dir, outp_dir,
             '--instance_prompt=a cartoon sticker on monotonous background',
             '--resolution=256', '--train_batch_size=1', '--gradient_accumulation_steps=1', '--learning_rate=5e-6',
             '--lr_scheduler=constant', '--lr_warmup_steps=0', '--max_train_steps=500'], check=True, text=True)
        print(torch.cuda.memory_allocated(device="cuda:1"))
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated(device="cuda:1"))

        bot.send_message(message.from_user.id, "Начинаю генерацию изображения.")
    except Exception as e:
        print("78", e)
        bot.send_message(message.from_user.id, "На сервере сейчас проблемы. Попробуйте отправить стикерпак позднее.")


def generate_prompt(message, image_path):
    print(f"GENERATE PROMPT FOR USER {message.from_user.id}...")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda:1")

        init_image = Image.open(image_path).convert("RGB")
        init_image = init_image.resize((256, 256))
        text = ""

        inputs = processor(init_image, text, return_tensors="pt").to("cuda:1")
        out = model.generate(**inputs)
        prompt = "a cartoon sticker of " + processor.decode(out[0],
                                                            skip_special_tokens=True) + " on monotonous background"

        return prompt
    except Exception as e:
        print("98", e)
        bot.send_message(message.from_user.id, "На сервере сейчас проблемы. Попробуйте позднее.")


def generate_sticker(message, image_path):
    global USER_MAP

    print(USER_MAP)
    print(f"GENERATING STICKER FOR USER {message.from_user.id}...")

    img_name = USER_MAP[message.from_user.id]["img_name"]
    os.makedirs(f'result_photo/{message.from_user.id}', exist_ok=True)

    path_to_model = f"tuned_model_sticker_pack/{message.from_user.id}"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(path_to_model, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    init_image = Image.open(image_path).convert("RGB")
    init_image = init_image.resize((256, 256))

    prompt = generate_prompt(message, image_path)

    result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
    result.save(f'result_photo/{message.from_user.id}/{img_name}.png')

    filename1 = f'result_photo/{message.from_user.id}/{img_name}.png'
    filename2 = f'result_photo/{message.from_user.id}/{img_name}.webp'
    script = subprocess.run(['rembg', 'i', filename1, filename2], check=True, text=True)
    torch.cuda.empty_cache()


def send_sticker(message):
    global USER_MAP

    print(USER_MAP)
    print(f"SENDING STICKER FOR USER {message.from_user.id}...")

    os.makedirs(f'sticker/{message.from_user.id}', exist_ok=True)

    img_name = USER_MAP[message.from_user.id]["img_name"]
    img = Image.open(f'result_photo/{message.from_user.id}/{img_name}.webp')

    if img.size[0] > img.size[1]:
        x = 512
        y = int(512 * img.size[1] / img.size[0])
    else:
        x = int(512 * img.size[0] / img.size[1])
        y = 512
    img = img.resize((x, y))

    image = np.asarray(img)
    if image.sum() == 0:
        bot.send_message(message.chat.id, "Произошла ошибка в генерации, попробуйте еще раз.")
    else:
        print('WITHOUT NSWF')
        img.save(f'sticker/{message.from_user.id}/{img_name}.png')

        bot.send_photo(message.chat.id, open(f"sticker/{message.from_user.id}/{img_name}.png", "rb"))

        shutil.rmtree(f'photo/{message.from_user.id}')
        shutil.rmtree(f'result_photo/{message.from_user.id}')

        markup = types.ReplyKeyboardMarkup(row_width=2)
        yes = types.KeyboardButton('добавить в стикерпак')
        no = types.KeyboardButton('не добавлять')
        markup.add(yes, no)
        mess = 'Хотите добавить картинку в стикерпак?'

        bot.send_message(message.chat.id, mess,
                         parse_mode='html',
                         reply_markup=markup)


def test(message):
    markup = types.ReplyKeyboardMarkup(row_width=2)
    yes = types.KeyboardButton('сделать стикер к новому стикерпаку')
    no = types.KeyboardButton('продолжить создавать по прошлому стикерпаку')
    markup.add(yes, no)
    mess = 'Хотите создать новый стикер по другому стикерпаку?'
    bot.send_message(message.chat.id, mess,
                     parse_mode='html',
                     reply_markup=markup)


@bot.message_handler(content_types=['photo'])
def get_user_photo(message):
    global USER_MAP

    print(USER_MAP)
    print(f"GET PHOTO FOR USER {message.from_user.id}...")

    try:
        sticker_set_name = USER_MAP[message.from_user.id]["sticker_set_name"]
        os.makedirs(f'photo/{message.from_user.id}', exist_ok=True)

        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        USER_MAP[message.from_user.id]["img_name"] = message.photo[1].file_id
        img_name = USER_MAP[message.from_user.id]["img_name"]

        src = f'photo/{message.from_user.id}/{img_name}.png'
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.reply_to(message,
                     "Фото добавлено. Начинаю обработку изображения. Процесс займет не более 5 минут. Пожалуйста, подождите")

        save = USER_MAP[message.from_user.id]["save"]
        print(save)
        if save:
            tune_dreambooth(message)
        generate_sticker(message, image_path=src)
        send_sticker(message)
    except Exception as e:
        print("203", e)
        bot.send_message(message.chat.id, "Сначала отправьте интересующий Вас стикерпак.")


def add_to_sticker_set(message):
    global USER_MAP

    print(USER_MAP)
    print(f"ADD STICKER FOR USER {message.from_user.id}...")

    sticker_set_name = f"s{message.from_user.id}_by_img2sticker_bot"
    img_name = USER_MAP[message.from_user.id]["img_name"]
    try:
        bot.get_sticker_set(sticker_set_name)
        image = open(f"sticker/{message.from_user.id}/{img_name}.png", "rb")
        bot.add_sticker_to_set(message.from_user.id, sticker_set_name, emojis=['🌳'],
                               png_sticker=image)
    except Exception as e:
        print("218", e)
        image = open(f"sticker/{message.from_user.id}/{img_name}.png", "rb")
        bot.create_new_sticker_set(user_id=message.from_user.id, name=sticker_set_name, title='your stickers',
                                   png_sticker=image,
                                   emojis=['🌳'])

    os.remove(f'sticker/{message.from_user.id}/{img_name}.png')

    sticker_set_link = f"t.me/addstickers/{sticker_set_name}"
    sticker_set = bot.get_sticker_set(sticker_set_name)
    bot.send_message(
        message.chat.id, sticker_set_link
    )
    bot.send_sticker(message.chat.id, sticker_set.stickers[-1].file_id)


@bot.message_handler(content_types=["sticker"])
def get_sticker_pack(message):
    global USER_MAP

    print(USER_MAP)
    print(f"GET STICKER PACK FOR USER {message.from_user.id}...")

    try:
        os.makedirs(f"sticker_pack/{message.from_user.id}", exist_ok=True)

        USER_MAP[message.from_user.id]["sticker_set_name"] = message.sticker.set_name
        bot.send_message(message.from_user.id, "Начинаю обрабатывать запрос. Пожалуйста, подождите")
        parce_sticker_set(message, message.sticker.set_name)
    except Exception as e:
        print("254", e)
        bot.send_message(message.chat.id, "Нажмите на кнопку СТАРТ.")


@bot.message_handler(commands=['start'])
def start(message):
    global USER_MAP
    print(USER_MAP)

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1)
    sticker = types.KeyboardButton('Сделать стикер')
    creators = types.KeyboardButton('Создатели')
    donate = types.KeyboardButton('Донат')
    markup.add(sticker, creators, donate)

    if message.from_user.id not in USER_MAP:
        USER["save"] = 1
        USER["sticker_set_name"] = ""
        USER["img_name"] = ""
        USER_MAP[message.from_user.id] = USER

    mess = f'Привет, {message.from_user.first_name}!! Чтобы создать свой стикер нажми кнопку "создать стикер" и следуй дальнейшим инструкциям'
    rules = open("rules.txt").read()
    bot.send_message(message.chat.id, rules)
    bot.send_message(message.chat.id, mess,
                     parse_mode='html',
                     reply_markup=markup)


@bot.message_handler()
def get_user_text(message):
    global USER_MAP

    if message.from_user.id not in USER_MAP:
        USER_MAP[message.from_user.id] = {}
        USER_MAP[message.from_user.id]["save"] = 1
        USER_MAP[message.from_user.id]["sticker_set_name"] = ""
        USER_MAP[message.from_user.id]["img_name"] = ""

    print(USER_MAP)
    if message.text == 'id':
        bot.send_message(message.chat.id, f'ваш ID {message.from_user.id}')
    elif message.text == 'добавить в стикерпак':
        add_to_sticker_set(message)
        bot.send_message(message.chat.id, "Готово!!")
        test(message)
    elif message.text == 'не добавлять':
        bot.send_message(message.chat.id, "Я Вас понял!!")
        test(message)
    elif message.text == 'сделать стикер к новому стикерпаку':
        USER_MAP[message.from_user.id]["save"] = 1
        bot.send_message(message.chat.id, "Пожалуйста, отправьте интересующий Вас стикерпак")
    elif message.text == 'продолжить создавать по прошлому стикерпаку':
        bot.send_message(message.chat.id, "Пожалуйста, отправьте ваше фото")
        USER_MAP[message.from_user.id]["save"] = 0
        print(USER_MAP[message.from_user.id]["save"])
    elif message.text == 'Сделать стикер':
        bot.send_message(message.chat.id, "Пожалуйста, отправьте интересующий Вас стикерпак")
    elif message.text == 'Донат':
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton('Донат',
                                              url='https://yandex.ru/images/search?text=%D0%BA%D0%B0%D1%80%D1%82%D0%B8%D0%BD%D0%BA%D0%B0+%D0%BF%D0%BE%D0%BD%D1%87%D0%B8%D0%BA%D0%B0+%D0%BD%D0%B0%D1%80%D0%B8%D1%81%D0%BE%D0%B2%D0%B0%D0%BD%D0%BD%D0%BE%D0%B3%D0%BE&img_url=http%3A%2F%2Fgas-kvas.com%2Fuploads%2Fposts%2F2023-01%2F1673578179_gas-kvas-com-p-risunok-detskii-ponchik-1.png&pos=0&rpt=simage&stype=image&lr=213&parent-reqid=1683015245087223-13411368779784305621-balancer-l7leveler-kubr-yp-sas-60-BAL-4327&source=serp'))
        bot.send_message(message.chat.id, 'Донат', reply_markup=markup)
    elif message.text == 'Создатели':
        bot.send_message(message.chat.id,
                         "Команда крутых ребят: \n Тася(@misshimichka) \n Александр(@alexsm1213)  \n Дима(@Something_is_going_wrong) \n Игорь(@Igk7p) \n Яна(@poltorq)")


bot.polling(none_stop=True)