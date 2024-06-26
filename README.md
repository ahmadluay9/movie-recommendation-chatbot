# Movie Recommendation ChatBot
This project aims to develop an movie / tv show recommendation chatbot that returns a list of recommended movie / tv show based on users preferences.

## Application Demo
[Application Demo](https://movie-recommendation-chatbot-ahmadluay9.streamlit.app/)

![movie-recommendation-gif](https://github.com/ahmadluay9/movie-recommendation-chatbot/assets/123846438/7727838e-e2a8-496a-b49c-cbb114727154)


## File Explanation
This repository consists of several files :
```
    ┌── .gitignore
    ├── README.md
    ├── app.py
    ├── notebook.ipynb
    ├── requirements.txt
    └── utils.py
```

- `gitignore`: Specifies intentionally untracked files to ignore. It is used by Git to determine which files and directories to ignore before committing code.

- `README.md`: A markdown file that typically contains an introduction and documentation for the project, including how to set up, run, and use the application.

- `app.py`: This file is the main script for the frontend of the application and is developed using the Streamlit framework.

- `notebook.ipynb`: A Jupyter Notebook file. This is commonly used for interactive development and testing, allowing for both code execution and rich text documentation.

- `requirements.txt`: A file listing the project's dependencies. It is used by pip to install all the necessary packages specified in this file.

- `utils.py`: A script containing utility functions or helper methods used by the main application

## Setup
1. Create a `.env` file containing the API key as below:
```
    OPENAI_API_KEY=
    TMDB_API_KEY=
```
2. Run streamlit
```
streamlit run app.py
```
3. Open http://localhost:8501
