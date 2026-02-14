delete from processed_accidents
where "Severity" is null
or "Start_Lat" is null
or "Start_Lng" is null;

update processed_accidents
set "Weather_Condition" = 'Unknown'
where "Weather_Condition" IS NULL;

SELECT 
    COUNT(*) - COUNT("Severity") AS missing_severity,
    COUNT(*) - COUNT("Weather_Condition") AS missing_weather,
    COUNT(*) - COUNT("Start_Time") AS missing_time,
    COUNT(*) - COUNT("Start_Lat") AS missing_location
FROM processed_accidents;

DELETE FROM processed_accidents a USING (
    SELECT MIN(ctid) as ctid, "Start_Time", "Start_Lat", "Start_Lng"
    FROM processed_accidents 
    GROUP BY "Start_Time", "Start_Lat", "Start_Lng"
    HAVING COUNT(*) > 1
) b
WHERE a."Start_Time" = b."Start_Time" 
AND a."Start_Lat" = b."Start_Lat" 
AND a."Start_Lng" = b."Start_Lng" 
AND a.ctid <> b.ctid;

SELECT count(*) FROM processed_accidents;

'''Initially there were 7728494 records. After deleting the duplicates 7315834 records left. Successfully deleted 412,560 duplicate records'''
