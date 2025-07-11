# Use a working Jupyter base image
FROM jupyter/base-notebook:python-3.11

# Set working directory
WORKDIR /home/jovyan/app

# Copy everything into the container
COPY . /home/jovyan/app

# Install Voilà and required packages
RUN pip install --no-cache-dir voila \
    leafmap \
    ipywidgets \
    scikit-learn \
    rasterio \
    pandas \
    numpy \
    matplotlib

# Enable widgets
# RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix

# Set Voilà as the default entrypoint
CMD ["voila", "--port=8888", "--no-browser", "deforestation-analysis.ipynb"]
