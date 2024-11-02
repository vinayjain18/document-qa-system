# 📚 Intelligent Document Q&A System

Welcome to the Intelligent Document Q&A System, a powerful tool designed to process PDF documents and provide concise answers to user queries based on the content of those documents. This application leverages advanced language models and document processing techniques to deliver accurate and contextually relevant responses.

## Features

- **PDF Document Processing**: Upload and process PDF documents to extract and store content for querying.
- **Contextual Q&A**: Ask questions and receive answers based solely on the content of the uploaded documents.
- **Chat History**: Maintain a history of questions and answers for easy reference.
- **Streamlit Interface**: User-friendly web interface built with Streamlit for seamless interaction.

## Technologies Used

- **Langchain**: For document processing and language model integration.
- **OpenAI GPT-4o**: Utilized for generating responses to user queries.
- **Streamlit**: Provides a simple and interactive web interface.
- **Chroma**: Used for storing and retrieving document embeddings.
- **PyPDFLoader**: For loading and parsing PDF documents.
- **RecursiveCharacterTextSplitter**: Splits documents into manageable chunks for processing.

## Installation

To set up the application locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vinayjain18/document-qa-system.git
   cd document-qa-system
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   Set your OpenAI API key in the environment:
   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   export OPENAI_MODEL='gpt-4o-mini'
   ```

4. **Run the Application**:
   Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload Documents**: Use the sidebar to upload PDF documents. Ensure each file is under 10MB.
2. **Process Documents**: Click the "Process Documents" button to extract and store document content.
3. **Ask Questions**: Once documents are processed, enter your question in the text input field.
4. **View Answers**: The system will provide answers based on the document content. Review the chat history for past interactions.

## Configuration

- **CHROMA_PATH**: Directory path for storing document embeddings.
- **MAX_HISTORY_LENGTH**: Maximum number of interactions stored in chat history.

## Logging

The application uses Python's built-in logging module to log information and errors. Logs are output to the console for easy monitoring.

## Future Use-cases:
- **Integration with other NLP models**: Integrate with other NLP models like BERT.
- **Support for multiple languages**: Add support for multiple languages.
- **Enhanced user interface**: Improve the user interface for better user experience.
- **Document categorization**: Implement document categorization based on content.
- **Other Document types**: Implement other types of documents like docs, html, xml, and many more.
- **Customer Support**: Implement it for creating Chatbots that uses your document to give precise answers.

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact [vinayjain449@gmail.com](mailto:vinayjain449@gmail.com).

---

Thank you for using the Intelligent Document Q&A System! We hope it enhances your document processing and querying experience.