# Carregar o python usado no desenvolvimento da aplicação
FROM python:3.12-slim

# Set the working directory
# Equivalente ao cd mlops_project, para acessar os arquivos da aplicação
WORKDIR /mlops_project

# Copy the entire project
# A partir da raiz do projeto, copiar todos os arquivos para dentro do container na pasta mlops_project
# O que estiver na .dockerignore será ignorado
COPY . .

# Install the package with dependencies
RUN pip install .

# Expose the port gunicorn will listen on
EXPOSE 5001

# Run gunicorn
# O gunicorn é mais seguro que o flask e é recomendado para produção
# O comando abaixo inicia o gunicorn, vinculando-o a todas as interfaces de rede na
#CMD ["gunicorn", "--bind=0.0.0.0:5001", "app.main:app"]
CMD ["gunicorn", "--workers=1", "--bind=0.0.0.0:5001", "app.main:app"]
