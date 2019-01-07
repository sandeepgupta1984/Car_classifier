FROM continuumio/anaconda3:4.4.0
WORKDIR .
COPY ./app .
EXPOSE 7000
RUN wget --quiet https://storage.googleapis.com/fastai_in_production/ImgItemList.pth && \
    apt-get update && apt install -y libgl1-mesa-glx && \
    conda install cymem && \
    conda install regex && \
    conda install spacy && \
    conda install thinc && \
    conda install -c fastai fastai==1.0.34 && \
    conda install -c pytorch pytorch torchvision && \
    pip install -r requirements.txt && \
    mv ImgItemList.pth /src_cars/models