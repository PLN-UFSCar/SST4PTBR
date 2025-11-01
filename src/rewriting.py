import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

punctuation_marks = set(string.punctuation)
stopwords_set = set(stopwords.words('portuguese'))

def sentences(text):
    return sent_tokenize(text, language='portuguese')

def words(sentence):
    tokenized_words = word_tokenize(sentence, language='portuguese')
    return [
        word for word in tokenized_words
        if word.lower() not in stopwords_set and word not in punctuation_marks and word.isalpha()
    ]

def generate_prompt(original_text, ironic_sentences, ambiguous_words_per_sentence):
    """
    Generates an optimized prompt to rewrite texts with irony and ambiguity.

    Args:
        original_text: Complete text to be rewritten
        ironic_sentences: List of sentences that contain irony
        ambiguous_words_per_sentence: Dictionary {sentence: [[word, specific_meaning]]}
            Where each item in the inner list is the word and its unique meaning already
            suggested by contextual analysis.

    Returns:
        String containing the structured and complete prompt.
    """

    # 1. CLARITY OF OBJECTIVE AND CONTEXT/PERSONA
    context = """Você é um assistente de reescrita de texto para pessoas com Transtorno do Espectro Autista (TEA). Seu objetivo principal é garantir que o texto seja CLARO, DIRETO E LITERAL, removendo QUALQUER ironia, sarcasmo, duplo sentido, linguagem figurada ou ambiguidade. Use um tom neutro e factual."""

    # 2. EXPLICIT INSTRUCTIONS AND HARD RESTRICTIONS (WITH VALIDATION)
    instructions = """
    INSTRUÇÕES E REGRAS ESSENCIAIS:

    1. **REMOÇÃO DE IRONIA:**
        - Substitua frases irônicas por suas mensagens LITERAIS e DIRETAS.
        - EXEMPLO: 'Que dia maravilhoso' (sarcástico) -> 'O dia foi ruim' (direto).

    2. **RESOLUÇÃO DE AMBIGUIDADE COM VALIDAÇÃO CONTEXTUAL:**
        - Para as palavras listadas abaixo, uma análise contextual prévia já sugeriu um sentido específico.
        - **SUA PRIMEIRA TAREFA É VALIDAR SE ESSA SUGESTÃO DE SENTIDO ESTÁ CORRETA.**
        - **SE A SUGESTÃO ESTIVER CORRETA:** Use-a para reescrever a frase de forma explícita e clara, eliminando a ambiguidade.
            - EXEMPLO: 'O banco estava fechado.' (Sugestão contextual: 'banco' -> instituição financeira) -> 'A agência bancária estava fechada.'
        - **SE VOCÊ, COMO UM MODELO DE LINGUAGEM SUPERIOR, JULGAR QUE A SUGESTÃO ESTÁ ERRADA ou incompleta:** Ignore a sugestão e use seu próprio entendimento do contexto para clarificar a frase da melhor forma possível.

    3. **CLAREZA E SIMPLICIDADE GERAL:**
        - Use frases curtas e objetivas. Evite metáforas, ditados populares e gírias.

    4. **PRESERVAÇÃO DO CONTEÚDO ORIGINAL:**
        - Mantenha TODAS as informações essenciais. NÃO remova ou adicione fatos.

    5. **RESTRIÇÃO CHAVE: NÃO ALTERE PARTES DO TEXTO QUE JÁ SÃO CLARAS.**
       APENAS modifique os segmentos identificados ou que você julgue problemáticos.
    """

    # 3. CONTEXT/SUPPORT DATA: ELEMENTS FOR ANALYSIS
    problematic_elements = ""

    if ironic_sentences or any(ambiguous_words_per_sentence.values()):
        problematic_elements += f"\n**ELEMENTOS ESPECÍFICOS PARA SUA ANÁLISE:**\n"

    if ironic_sentences:
        problematic_elements += f"\n**FRASES COM POTENCIAL IRONIA:**\n"
        for i, sentence in enumerate(ironic_sentences, 1):
            problematic_elements += f"{i}. \"{sentence}\"\n"

    if any(ambiguous_words_per_sentence.values()):
        problematic_elements += f"\n**PALAVRAS AMBÍGUAS COM SUGESTÃO CONTEXTUAL PARA VALIDAÇÃO:**\n"
        for sentence, words_info_list in ambiguous_words_per_sentence.items():
            if words_info_list:
                formatted_words = []
                for word_info in words_info_list:
                    word = word_info[0]
                    suggested_meaning = word_info[1]
                    # Formats the output to show the word and its unique suggested contextual meaning.
                    formatted_words.append(f"'{word}' (sentido contextual sugerido: {suggested_meaning})")
                problematic_elements += f"- Na frase \"{sentence}\", valide a seguinte sugestão: {', '.join(formatted_words)}\n"

    # 4. EXAMPLES (FEW-SHOT) - Aligned with the new logic
    examples = """
    **EXEMPLOS DE REESCRITA (ENTRADA -> SAÍDA DIRETAS):**

    **Exemplo 1 (Ironia):**
    ENTRADA: "Nossa, que dia ótimo! Esqueci a carteira em casa."
    SAÍDA: "Foi um dia ruim. Eu esqueci minha carteira em casa."

    **Exemplo 2 (Ambiguidade Validada):**
    ENTRADA: "Comprei uma manga para o lanche." (Sugestão contextual: 'manga' -> fruta)
    SAÍDA: "Comprei uma fruta manga para o lanche."

    **Exemplo 3 (Ambiguidade onde o LLM pode refinar a sugestão):**
    ENTRADA: "A manga da camisa rasgou." (Sugestão contextual: 'manga' -> parte de uma roupa)
    SAÍDA: "A parte da camisa que cobre o braço, a manga, rasgou."

    **Exemplo 4 (Preservar clareza, não alterar):**
    ENTRADA: "A água é essencial para a vida."
    SAÍDA: "A água é essencial para a vida."
    """

    # 5. FINAL PROMPT STRUCTURE AND FINAL OUTPUT INSTRUCTION
    prompt = f"""{context}

    {instructions}

    {examples}

    {problematic_elements}

    **TEXTO ORIGINAL A SER REESCRITO:**
    \"{original_text}\"

    **TAREFA FINAL:**
    Com base nas **INSTRUÇÕES E REGRAS ESSENCIAIS** e nos **EXEMPLOS**, reescreva o **TEXTO ORIGINAL A SER REESCRITO**.
    Lembre-se de **validar o contexto** das ambiguidades sugeridas antes de fazer qualquer alteração.
    **FORNEÇA APENAS O TEXTO REESCRITO FINAL, SEM QUALQUER TEXTO INTRODUTÓRIO OU EXPLICATIVO.**

    **TEXTO REESCRITO FINAL:**"""

    return prompt