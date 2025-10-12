import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
#import lmstudio as lms

pontuacoes = set(string.punctuation)
stopwords_set = set(stopwords.words('portuguese'))

def frases(texto):
    return sent_tokenize(texto, language='portuguese')

def palavras(frase):
    palavras_tokenizadas = word_tokenize(frase, language='portuguese')
    return [
        palavra for palavra in palavras_tokenizadas
        if palavra.lower() not in stopwords_set and palavra not in pontuacoes and palavra.isalpha()
    ]

def gerarPrompt(texto_original, frases_ironicas, palavras_ambiguas_por_frase):
    """
    Gera um prompt otimizado para reescrever textos com ironia e ambiguidade,
    preparado para a lógica da futura função de detecção que identifica um
    sentido específico para palavras ambíguas com base no contexto.

    Args:
        texto_original: Texto completo a ser reescrito
        frases_ironicas: Lista de frases que contêm ironia
        palavras_ambiguas_por_frase: Dicionário {frase: [[palavra, sentido_especifico]]}
            Onde cada item na lista interna é a palavra e seu sentido único já
            sugerido pela análise contextual.

    Returns:
        String contendo o prompt estruturado e completo.
    """

    # 1. CLAREZA DE OBJETIVO E CONTEXTO/PERSONA
    contexto = """Você é um assistente de reescrita de texto para pessoas com Transtorno do Espectro Autista (TEA). Seu objetivo principal é garantir que o texto seja CLARO, DIRETO E LITERAL, removendo QUALQUER ironia, sarcasmo, duplo sentido, linguagem figurada ou ambiguidade. Use um tom neutro e factual."""

    # 2. INSTRUÇÕES EXPLÍCITAS E RESTRICOES DURAS (COM VALIDAÇÃO)
    instrucoes = """
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

    # 3. CONTEXTO/DADOS DE APOIO: ELEMENTOS PARA ANÁLISE
    elementos_problematicos = ""

    if frases_ironicas or any(palavras_ambiguas_por_frase.values()):
        elementos_problematicos += f"\n**ELEMENTOS ESPECÍFICOS PARA SUA ANÁLISE:**\n"

    if frases_ironicas:
        elementos_problematicos += f"\n**FRASES COM POTENCIAL IRONIA:**\n"
        for i, frase in enumerate(frases_ironicas, 1):
            elementos_problematicos += f"{i}. \"{frase}\"\n"

    if any(palavras_ambiguas_por_frase.values()):
        elementos_problematicos += f"\n**PALAVRAS AMBÍGUAS COM SUGESTÃO CONTEXTUAL PARA VALIDAÇÃO:**\n"
        for frase, palavras_info_lista in palavras_ambiguas_por_frase.items():
            if palavras_info_lista:
                palavras_formatadas = []
                for palavra_info in palavras_info_lista:
                    palavra = palavra_info[0]
                    sentido_sugerido = palavra_info[1]
                    # Formata a saída para mostrar a palavra e seu sentido contextual único sugerido.
                    palavras_formatadas.append(f"'{palavra}' (sentido contextual sugerido: {sentido_sugerido})")
                elementos_problematicos += f"- Na frase \"{frase}\", valide a seguinte sugestão: {', '.join(palavras_formatadas)}\n"

    # 4. EXEMPLOS (FEW-SHOT) - Alinhados com a nova lógica
    exemplos = """
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

    # 5. ESTRUTURA FINAL DO PROMPT E INSTRUÇÃO DE SAÍDA FINAL
    prompt = f"""{contexto}

    {instrucoes}

    {exemplos}

    {elementos_problematicos}

    **TEXTO ORIGINAL A SER REESCRITO:**
    \"{texto_original}\"

    **TAREFA FINAL:**
    Com base nas **INSTRUÇÕES E REGRAS ESSENCIAIS** e nos **EXEMPLOS**, reescreva o **TEXTO ORIGINAL A SER REESCRITO**.
    Lembre-se de **validar o contexto** das ambiguidades sugeridas antes de fazer qualquer alteração.
    **FORNEÇA APENAS O TEXTO REESCRITO FINAL, SEM QUALQUER TEXTO INTRODUTÓRIO OU EXPLICATIVO.**

    **TEXTO REESCRITO FINAL:**"""

    return prompt
