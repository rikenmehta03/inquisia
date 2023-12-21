FROM ubuntu:22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y python3.10 python3-pip 

RUN ln -s /usr/bin/python3.10 /usr/local/bin/python

RUN PIP_INSTALL="python -m pip install --upgrade --no-cache-dir --retries 10 --timeout 60" && \
    $PIP_INSTALL \
    pymilvus \
    langchain \
    litellm

WORKDIR /inquisia
COPY . /inquisia

RUN pip install .

CMD ["/bin/bash"]