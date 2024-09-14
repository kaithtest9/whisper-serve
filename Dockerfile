FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y \
    nginx

COPY ./nginx.conf /etc/nginx/nginx.conf
COPY index.html /var/www/html/index.html

CMD ["nginx", "-g", "daemon off;"]