CURRENT_LANG = "ua"

TRANSLATIONS = {
    "en": {
        "Конструктор гребель": "Dam Constructor",

        " Метод розрахунку ": " Calculation Method ",
        "Скінченні елементи": "Finite Elements",
        "Скінченні різниці": "Finite Differences",

        " Геометрія області ": " Domain Geometry ",
        "Ширина області (м)": "Domain width (m)",
        "Глибина області (м)": "Domain depth (m)",

        " Геометрія греблі ": " Dam Geometry ",
        "Початок греблі (м)": "Dam start (m)",
        "Кінець греблі (м)": "Dam end (m)",
        "Глибина основи (м)": "Foundation depth (m)",

        " Шпунти ": " Sheet Piles ",
        "+ Додати шпунт": "+ Add sheet pile",
        "Розташування:": "Location:",
        "Довжина:": "Length:",

        " Фізичні властивості ": " Physical Properties ",
        "Коеф. фільтрації": "Filtration coeff.",
        "Коеф. дифузії": "Diffusion coeff.",
        "Пористість": "Porosity",
        "Напір лівого б'єфу (м)": "Left upstream head (m)",
        "Напір правого б'єфу (м)": "Right downstream head (m)",
        "Період (діб)": "Period (days)",

        " Властивості забруднення ": " Pollution Properties ",
        "Початок (м)": "Start (m)",
        "Кінець (м)": "End (m)",
        "Концентрація (0-1)": "Concentration (0-1)",
        "Постійне джерело": "Continuous source",
        "Тимчасове джерело": "Temporary source",
        "Тривалість забруднення (діб):": "Pollution duration (days):",

        "РОЗРАХУВАТИ": "CALCULATE",
        "ЗУПИНИТИ РОЗРАХУНОК": "STOP CALCULATION",
        "Скинути налаштування": "Reset settings",
        "Готовий": "Ready",
        "Налаштування скинуто": "Settings reset",
        "Йде розрахунок...": "Calculating...",
        "Зупиняємо...": "Stopping...",
        "Розрахунок зупинено": "Calculation stopped",
        "Готово": "Done",

        "Відображати сітку": "Show mesh",
        "Винести в окреме вікно": "Detach to new window",
        "Закрити": "Close",
        "Згорнути": "Attach",
        "День": "Day",
        "Напір": "Hydraulic Head",

        "Розрахунок зупинено користувачем": "Calculation stopped by user",
        "Помилка: ": "Error: ",
        "Час розрахунку: ": "Calculation time: ",
        " с": " s",
        "Забруднення: День ": "Pollution: Day ",
        "Рішення ": "Solution ",
        "МСЕ": "FEM",
        "МСР": "FDM"
    }
}


def tr(text):
    if CURRENT_LANG == "ua":
        return text
    return TRANSLATIONS.get(CURRENT_LANG, {}).get(text, text)


def set_lang(lang):
    global CURRENT_LANG
    CURRENT_LANG = lang