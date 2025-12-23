import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

load_dotenv()


def _get_api_key() -> str:
    # Read from environment variable loaded via .env
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return api_key


def build_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    # Use your working Gemini model
    return ChatGoogleGenerativeAI(
        google_api_key=_get_api_key(),
        model="gemini-2.5-flash",
        temperature=temperature,
    )


def build_slogan_chain(temperature: float = 0.2):
    """Build a slogan generation chain using prompt and LLM."""
    prompt = PromptTemplate.from_template(
        "Suggest a catchy T-shirt slogan about {topic}."
    )
    llm = build_llm(temperature)
    
    class SimpleChain:
        def __init__(self, llm, prompt):
            self.llm = llm
            self.prompt = prompt
        
        def run(self, inputs: dict):
            formatted_prompt = self.prompt.format(**inputs)
            response = self.llm.invoke(formatted_prompt)
            return response.content if hasattr(response, 'content') else str(response)
    
    return SimpleChain(llm, prompt)


def get_few_shot_db_chain(
    top_k: int = 5,
    temperature: float = 0.1,
    sample_rows_in_table_info: int = 3,
) -> SQLDatabaseChain:
    """
    Connect to the MySQL instance and create a SQLDatabaseChain.
    Credentials are read from environment variables:
      DB_USER (default: root)
      DB_PASSWORD (default: root)
      DB_HOST (default: localhost)
      DB_NAME (default: atliq_tshirts)
    """
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "root")
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("DB_NAME", "atliq_tshirts")

    uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=sample_rows_in_table_info)

    llm = build_llm(temperature)
    return SQLDatabaseChain.from_llm(llm, db, top_k=top_k, verbose=True)


def test_database_connection():
    """Test basic database connection without LLM."""
    try:
        db_user = os.getenv("DB_USER", "root")
        db_password = os.getenv("DB_PASSWORD", "root")
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "atliq_tshirts")
        
        uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
        print(f"Testing database connection to: {db_host}/{db_name}")
        
        db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=3)
        print("✓ Database connection successful!")
        print(f"\nDatabase tables: {db.get_usable_table_names()}")
        print(f"\nTable info:\n{db.table_info}")
        
        # Test a simple query
        result = db.run("SELECT COUNT(*) as total FROM t_shirts")
        print(f"\n✓ Test query successful! Total t-shirts: {result}")
        
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Database Connection")
    print("=" * 60)
    
    # Test database connection first
    if test_database_connection():
        print("\n" + "=" * 60)
        print("Testing SQLDatabaseChain with LLM")
        print("=" * 60)
        
        try:
            # Test with a simple question
            chain = get_few_shot_db_chain()
            question = "How many t-shirts are in the database?"
            print(f"\nQuestion: {question}")
            print("\nGenerating answer...")
            # SQLDatabaseChain can be called directly with a string
            response = chain(question)
            # Extract result from response dictionary
            result = response.get('result', response) if isinstance(response, dict) else response
            print(f"\nAnswer: {result}")
            
            # Test another question
            print("\n" + "-" * 60)
            question2 = "What are the available brands?"
            print(f"Question: {question2}")
            response2 = chain(question2)
            result2 = response2.get('result', response2) if isinstance(response2, dict) else response2
            print(f"Answer: {result2}")
            
            # Test a more complex question
            print("\n" + "-" * 60)
            question3 = "Show me Nike t-shirts that are available in size L"
            print(f"Question: {question3}")
            response3 = chain(question3)
            result3 = response3.get('result', response3) if isinstance(response3, dict) else response3
            print(f"Answer: {result3}")
            
            print("\n" + "=" * 60)
            print("✓ All database query tests completed successfully!")
            print("=" * 60)
        except Exception as e:
            print(f"\n✗ SQLDatabaseChain test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nPlease fix database connection issues before testing SQLDatabaseChain.")

