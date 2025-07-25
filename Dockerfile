FROM ubuntu:latest
LABEL authors="nikitamarshchonok"

ENTRYPOINT ["top", "-b"]