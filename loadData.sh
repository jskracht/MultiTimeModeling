#!/bin/bash

set -e
set -u

# Set these environmental variables to override them,
# but they have safe defaults.
export PGHOST=${PGHOST-localhost}
export PGPORT=${PGPORT-5432}
export PGDATABASE="Jesh"
export PGUSER="Jesh"
export PGPASSWORD=

RUN_PSQL="psql -X --set AUTOCOMMIT=on --set ON_ERROR_STOP=on "

COUNT=1
while read NAME; do

FILE="'$(sed -n "${COUNT}p" ./FRED/index)'"

echo $FILE
${RUN_PSQL} <<SQL
CREATE TABLE ${NAME} (date date, value numeric,  
  CONSTRAINT date_${NAME} PRIMARY KEY (date),
  CONSTRAINT date_${NAME} UNIQUE (date));
COPY ${NAME}(date, value) FROM $FILE DELIMITER ',' CSV HEADER;
SQL
COUNT=$((COUNT + 1))

done <./FRED/list