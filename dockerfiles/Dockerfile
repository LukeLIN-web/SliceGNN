#!/usr/bin/env -S sh -c 'docker build --rm -t torch_quiver:snapshot . -f $0 && docker run --rm -it torch_quiver:snapshot'

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Install PyG.
RUN CPATH=/usr/local/cuda/include:$CPATH && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && \
    DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

RUN pip install scipy==1.5.0

RUN pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html && \
    pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html && \
    pip install torch-geometric

WORKDIR /quiver
ADD . .
RUN pip install -v .

# Set the default command to python3.
CMD ["python3"]
