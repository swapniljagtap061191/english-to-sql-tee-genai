# English to SQL: T-Shirt Database Q&A

An intelligent natural language to SQL query converter built with **Google Gemini AI** and **LangChain**. This project enables users to query a MySQL database using plain English questions, automatically generating and executing SQL queries.

![Project Preview]
(nat_lang_streamlit_ui.png)

## 🌟 Features

- **Natural Language Processing**: Convert English questions to SQL queries automatically
- **Google Gemini AI Integration**: Powered by `gemini-2.5-flash` model for accurate query generation
- **MySQL Database Integration**: Seamlessly connects to MySQL databases
- **Streamlit Web Interface**: User-friendly web UI for interactive querying
- **Intelligent Query Generation**: Context-aware SQL query generation with proper table relationships

## 📋 Project Overview

**AtliQ Tees** is a T-shirt store that sells Adidas, Nike, Van Heusen, and Levi's t-shirts. The system allows store managers to ask questions in natural language, which are then converted to SQL queries and executed on the MySQL database.

### Example Questions:
- "How many white color Adidas t-shirts do we have left in stock?"
- "What are the available brands?"
- "Show me Nike t-shirts that are available in size L"
- "How much sales amount will be generated if we sell all small size Adidas shirts after discounts?"

## 🛠️ Tech Stack

- **LLM**: Google Gemini 2.5 Flash (`gemini-2.5-flash`)
- **Framework**: LangChain
- **Database**: MySQL
- **UI**: Streamlit
- **Python**: 3.x

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- MySQL Server installed and running
- Google API Key (from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/swapniljagtap061191/english-to-sql-tee-genai.git
   cd english-to-sql-tee-genai
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   DB_USER=root
   DB_PASSWORD=your_db_password
   DB_HOST=localhost
   DB_NAME=atliq_tshirts
   ```

5. **Set up the database**:
   - Open MySQL Workbench or your MySQL client
   - Run the SQL script: `database/db_creation_atliq_t_shirts.sql`
   - This will create the database and populate it with sample data

## 🚀 Usage

### Running the Streamlit App

```bash
streamlit run main.py
```

The web application will open in your browser at `http://localhost:8501`

### Testing the Database Connection

You can test the database connection and query functionality by running:

```bash
python langchain_helper.py
```

This will:
- Test the database connection
- Display table information
- Run sample queries using the LLM

## 📁 Project Structure

```
4_sqldb_tshirts/
│
├── main.py                  # Streamlit web application
├── langchain_helper.py      # LangChain integration and database chain setup
├── few_shots.py             # Few-shot learning examples (if applicable)
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not tracked in git)
├── .gitignore              # Git ignore file
├── README.md               # Original README file
├── readme_updated.md       # This updated README file
├── atliq_tees.png          # Project preview image
├── t_shirt_sales_llm.ipynb # Jupyter notebook with original exploration
│
└── database/
    └── db_creation_atliq_t_shirts.sql  # Database schema and sample data
```

## 🔧 Configuration

### Database Configuration

Update the `.env` file with your MySQL credentials:

```env
DB_USER=your_mysql_username
DB_PASSWORD=your_mysql_password
DB_HOST=localhost
DB_NAME=atliq_tshirts
```

### Model Configuration

The LLM model can be configured in `langchain_helper.py`:

```python
def build_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        google_api_key=_get_api_key(),
        model="gemini-2.5-flash",  # Change model here if needed
        temperature=temperature,
    )
```

## 📝 Sample Questions

Try asking questions like:

- "How many total t-shirts are left in stock?"
- "What are all the available brands?"
- "Show me all Nike t-shirts in size L"
- "How many t-shirts do we have for Nike in XS size and white color?"
- "What's the total price of inventory for all small size t-shirts?"
- "Which t-shirts have discounts available?"

## 🔍 How It Works

1. **User Input**: User enters a question in natural language through the Streamlit interface
2. **LLM Processing**: The Gemini model processes the question along with database schema information
3. **SQL Generation**: The LLM generates an appropriate SQL query based on the question
4. **Query Execution**: The generated SQL query is executed on the MySQL database
5. **Result Processing**: The results are processed and formatted by the LLM
6. **Response Display**: The final answer is displayed to the user

## 📚 Key Components

### `langchain_helper.py`

Contains the core functionality:
- `build_llm()`: Initializes the Google Gemini LLM
- `get_few_shot_db_chain()`: Creates the SQL database chain for querying
- `test_database_connection()`: Utility function to test database connectivity

### `main.py`

Streamlit application that provides:
- Text input for questions
- Query execution and result display
- Error handling and user feedback

## 🔄 Migration from Google Palm

This project was originally built with **Google Palm**, which has been deprecated. The current implementation uses **Google Gemini 2.5 Flash**, which is:
- The successor to Google Palm
- More powerful and accurate
- Better supported in modern LangChain versions
- Uses the same API key system

The functionality remains the same - users can still query the database using natural language, but with improved performance and reliability.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Google Gemini AI](https://ai.google.dev/)
- UI created with [Streamlit](https://streamlit.io/)

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: Make sure to keep your `.env` file secure and never commit it to version control. The `.gitignore` file is already configured to exclude it.

<<<<<<< HEAD
=======
- main.py: The main Streamlit application script.
- langchain_helper.py: This has all the langchain code
- requirements.txt: A list of required Python packages for the project.
- few_shots.py: Contains few shot prompts
- .env: Configuration file for storing your Google API key.


NOTE: This project is performed by Dhawal Patel (Youtube: https://www.youtube.com/watch?v=4wtrl4hnPT8)
As there was Google Palm API been used in the project, now is deprecated. I upgraded the project to use Google's newer model: Google Gemini
>>>>>>> 17f472295df4f258e7f8161aaa8f3a88817178e4
