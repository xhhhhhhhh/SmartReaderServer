FROM python:latest

WORKDIR /app

RUN pip install flask
RUN pip install google-generativeai
RUN pip install python-dotenv
RUN pip install pydantic-ai
RUN pip install langchain
RUN pip install chromadb

COPY server.py .
COPY .env .

EXPOSE 8080

CMD ["python", "server.py"]