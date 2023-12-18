FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN PIP_INSTALL="python -m pip install --upgrade --no-cache-dir --retries 10 --timeout 60" && \
    $PIP_INSTALL \
    pymilvus \
    transformers \
    bitsandbytes \
    peft \
    langchain \
    sentence-transformers \
    litellm

WORKDIR /inquisia
COPY . /inquisia

RUN pip install .

ENTRYPOINT ["/bin/bash"]