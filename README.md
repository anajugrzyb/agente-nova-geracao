# 🎓 Assistente Virtual — Universidade Nova Geração

Agente inteligente baseado em **RAG (Retrieval-Augmented Generation)** que responde,
em linguagem natural, perguntas de alunos sobre documentos institucionais de uma
universidade fictícia: **Calendário Acadêmico, Matrícula, TCC, Biblioteca, Estágio
e Regulamento Geral**.

Projeto desenvolvido para portfólio, aplicando arquitetura de
aplicações de IA Generativa: ingestão de documentos, indexação
vetorial, recuperação semântica e geração de respostas fundamentadas.

>  **Importante:** o agente responde **exclusivamente** com base no conteúdo dos
> documentos indexados. Quando a informação não está presente, ele responde:
> *"Essa informação não está presente na documentação da universidade."*

---

## 🛠️ Tecnologias

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3.11+ |
| Orquestração de LLM | [LangChain](https://python.langchain.com/) |
| Orquestração de fluxo/agente | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM | [GROQ](https://groq.com/) (gratuito — `llama-3.1-8b-instant`) |
| Embeddings | [HuggingFace Sentence-Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`, local, gratuito) |
| Banco vetorial | [FAISS](https://faiss.ai/) (`faiss-cpu`) |
| Interface | [Streamlit](https://streamlit.io/) |
| Dados / utilitários | Pandas, python-dotenv |
| Leitura de PDF | `pypdf` |

Todas as tecnologias utilizadas possuem camada gratuita — nenhum serviço pago é
necessário para rodar o projeto.

---

## 📂 Estrutura do Projeto

```
projeto/
│
├── app.py                 
├── ingest.py              
├── rag.py                  
├── prompts.py               
├── config.py                
├── requirements.txt          
├── README.md                 
├── .env                     
├── .gitignore
│
├── .streamlit/
│   └── config.toml            
│
├── data/                      
│   ├── calendario.pdf
│   ├── matricula.pdf
│   ├── biblioteca.pdf
│   ├── estagio.pdf
│   ├── tcc.pdf
│   └── regulamento.pdf
│
├── vectorstore/               
│
└── utils/
    ├── __init__.py
    ├── logger.py               
    └── helpers.py               
```

---

## ⚙️ Instalação

### 1. Pré-requisitos
- Python 3.11 ou superior
- Conta gratuita na [GROQ Console](https://console.groq.com/) para gerar sua `GROQ_API_KEY`

### 2. Clone o repositório
```bash
git clone https://github.com/anajugrzyb/agente-nova-geracao.git
cd agente-nova-geracao
```

### 3. Crie e ative um ambiente virtual
```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4. Instale as dependências
```bash
pip install -r requirements.txt
```

### 5. Configure as variáveis de ambiente
```bash
cp .env
```
Edite o arquivo `.env` e insira sua chave gratuita da GROQ:
```env
GROQ_API_KEY=sua_chave_groq_aqui
```

---

## ▶️ Como Executar

### Opção A — Ingestão via linha de comando (opcional)
Você pode gerar o índice vetorial antes de abrir a interface:
```bash
python ingest.py
```
Isso lê os PDFs em `data/`, gera os embeddings e salva o índice em `vectorstore/`.

### Opção B — Direto pela interface
Basta iniciar a aplicação; se não houver índice salvo, use o botão da sidebar:
```bash
streamlit run app.py
```
Acesse `http://localhost:8501` no navegador e clique em
**"🔄 Carregar / Reprocessar Documentos"** na barra lateral.

> Da segunda execução em diante, o índice salvo em `vectorstore/` é carregado
> automaticamente — sem necessidade de reprocessar os documentos.

---

## 💡 Exemplos de Uso

| Pergunta do aluno | Documento consultado |
|---|---|
| "Como faço minha matrícula?" | Matrícula |
| "Posso trancar uma disciplina?" | Matrícula |
| "Como renovar livros?" | Biblioteca |
| "Qual o prazo máximo de empréstimo?" | Biblioteca |
| "Como funciona o estágio obrigatório?" | Estágio |
| "Quantas horas de estágio preciso cumprir?" | Estágio |
| "Como entregar o TCC?" | TCC |
| "Qual a nota mínima para aprovação?" | TCC / Regulamento Geral |
| "Quantas faltas posso ter?" | Regulamento Geral |
| "Quando começam as férias?" | Calendário Acadêmico |
| "Como solicitar segunda chamada?" | Regulamento Geral / Calendário |
| "Quais documentos preciso para matrícula?" | Matrícula |

Exemplo de pergunta **fora do escopo** dos documentos:
> **Pergunta:** "Qual o valor da mensalidade do curso de Medicina?"
> **Resposta:** *"Essa informação não está presente na documentação da universidade."*

---

## 🌐 Aplicação Online

Acesse a versão publicada do projeto no Streamlit Community Cloud:

🔗 **[https://agente-nova-geracao.streamlit.app/](https://agente-nova-geracao.streamlit.app/)**

---

## 🎥 Demonstração em Vídeo

Confira o vídeo abaixo mostrando o funcionamento completo do agente, desde o
carregamento dos documentos até a resposta a perguntas dos alunos:

[![Assista à demonstração](https://drive.google.com/file/d/1dHZLQuO7hUMPu-YVZSEAISGoU9q-Ckkn/view?usp=sharing)

