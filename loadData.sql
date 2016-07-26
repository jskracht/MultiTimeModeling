CREATE TABLE timeseries (date date, value int,  
  CONSTRAINT date PRIMARY KEY (date),
  CONSTRAINT date UNIQUE (date));
COPY timeseries(date, value) FROM '/Users/Jesh/Documents/Project/FRED/4/4BIGEURORECD.csv' DELIMITER ',' CSV HEADER;

CREATE TABLE timeseries2 (date date, value int,  
  CONSTRAINT date2 PRIMARY KEY (date),
  CONSTRAINT date2 UNIQUE (date));
COPY timeseries2(date, value) FROM '/Users/Jesh/Documents/Project/FRED/4/4BIGEURORECDP.csv' DELIMITER ',' CSV HEADER;

CREATE TABLE timeseries3 (date date, value int,  
  CONSTRAINT date3 PRIMARY KEY (date),
  CONSTRAINT date3 UNIQUE (date));
COPY timeseries3(date, value) FROM '/Users/Jesh/Documents/Project/FRED/4/4BIGEUROREC.csv' DELIMITER ',' CSV HEADER;

ALTER TABLE timeseries ADD COLUMN value2 integer;
UPDATE timeseries
SET value2 = timeseries2.value
FROM timeseries2
WHERE timeseries.date = timeseries2.date;

ALTER TABLE timeseries ADD COLUMN value3 integer;
UPDATE timeseries
SET value3 = timeseries3.value
FROM timeseries3
WHERE timeseries.date = timeseries3.date;