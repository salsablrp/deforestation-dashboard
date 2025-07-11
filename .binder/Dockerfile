FROM jupyter/voila:python-3.11

COPY . /home/jovyan/app
WORKDIR /home/jovyan/app

# Install only what's needed
RUN pip install --no-cache-dir numpy pandas scikit-learn matplotlib leafmap ipywidgets xyzservices

EXPOSE 8888
CMD ["voila", "deforestation-dashboard.ipynb", "--port=8888", "--no-browser", "--VoilaConfiguration.enable_nbextensions=True"]
