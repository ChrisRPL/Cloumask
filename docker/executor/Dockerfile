# Cloumask Script Executor
# Isolated environment for running user-generated CV scripts

FROM python:3.11-slim

# Install CV dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless==4.9.0.80 \
    numpy==1.26.4 \
    pillow==10.3.0 \
    scipy==1.13.0 \
    scikit-image==0.23.2 \
    # Common data formats
    pandas==2.2.2 \
    # Image augmentation
    albumentations==1.4.4

# Create workspace and non-root user for security
WORKDIR /workspace
RUN useradd -m -s /bin/bash executor && \
    chown -R executor:executor /workspace

# Copy entrypoint script
COPY entrypoint.py /usr/local/bin/cloumask-entrypoint.py
RUN chmod +x /usr/local/bin/cloumask-entrypoint.py

# Switch to non-root user
USER executor

# Use Cloumask entrypoint wrapper
ENTRYPOINT ["python", "/usr/local/bin/cloumask-entrypoint.py"]
