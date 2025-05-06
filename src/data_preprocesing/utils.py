import re
from bs4 import BeautifulSoup  

'''
Функция для маскирования. Мы будем маскировать математические выражениея в тексте, чтобы
помочь модели лучше распознавать контекст вокруг математических выражений и сами математические
выражеия.
'''
def mask_math(text):
    # Preserve mathematical notation
    text = re.sub(r'\$(.*?)\$', r' [MATH] \1 [MATH] ', text)
    text = re.sub(r'\\\w+', lambda m: ' ' + m.group(0) + ' ', text)
    return text.strip()

def mask_latex(text):
    environments = [
        "equation", "equation*", "align", "align*", "multline",
        "multline*", "eqnarray", "eqnarray*"
    ]

    # patterns = [
    #     rf'\\begin\{{{env}\}}.*?\\end\{{{env}\}}' for env in environments
    # ]
    
    patterns = [
        r'\\begin{equation}.*?\\end{equation}',
        r'\\begin{equation\*}.*?\\end{equation\*}',
        r'\\begin{align}.*?\\end{align}',
        r'\\begin{align\*}.*?\\end{align\*}',
        r'\\begin{multline}.*?\\end{multline}',
        r'\\begin{multline\*}.*?\\end{multline\*}',
        r'\\begin{eqnarray}.*?\\end{eqnarray}',
        r'\\begin{eqnarray\*}.*?\\end{eqnarray\*}',
    ]

    patterns += [
        r'\$\$.*?\$\$',  # $$...$$
        r'(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)'  # одиночные $
    ]

    combined_pattern = '|'.join(f'({p})' for p in patterns)
    matches = list(re.finditer(combined_pattern, text, re.DOTALL))

    result = ""
    last_idx = 0

    for match in matches:
        start, end = match.span()
        result += text[last_idx:start]
        result += f"[LATEX_START]{text[start:end]}[LATEX_END]"
        last_idx = end

    result += text[last_idx:]
    return result

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # --- Базовая предобработка ---

    # Удаление HTML-тегов
    # text = BeautifulSoup(text, "html.parser").get_text()
    
    # вместо удаления html текста, мы можем его маскировать
    text = re.sub(r'<[^>]+>', '[HTML]', text)

    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление URL-ов
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # Удаление email-адресов
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)

    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    # --- Специальная предобработка ---

    # 7. Маскировка LaTeX
    text = mask_latex(text)  # см. функцию ниже

    # 8. Маскировка кода
    text = re.sub(r'```.*?```', '[CODE]', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '[CODE]', text)  # inline code

    # 9. Удаление markdown форматирования
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'^#+\s?', '', text, flags=re.MULTILINE)  # заголовки

    # 10. Нормализация чисел
    text = re.sub(r'\b\d+(\.\d+)?\b', '[NUM]', text)

    return text

SPECIAL_TOKENS = ['MATH', 'URL', '[EMAIL]', '[CODE]', '[NUM]']
