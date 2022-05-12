FROM python:slim

RUN pip install pandas numpy seaborn matplotlib openpyxl sklearn xgboost

COPY /data /data
COPY python.py .

ENTRYPOINT ["python", "python.py"]