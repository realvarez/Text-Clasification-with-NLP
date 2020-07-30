SELECT DISTINCT 
    "a"."name" AS "Area_Name", 
    "d"."name" AS "Dimension_Name",
    "ans"."id",
    "ans"."question_id",
    "ans"."answer" AS "Answer"
FROM "areas" "a", "dimensions" "d" , "questions" "q", "answers" "ans"
WHERE "a"."id" = "d"."area_id" AND "d"."id" = "q"."dimension_id" AND "q"."id" = "ans"."question_id" AND "ans"."answer" !=   '' AND strpos(answer, 'ecoss') > 0;


SELECT DISTINCT 
    "a"."name" AS "Area_Name", 
    "d"."name" AS "Dimension_Name",
    "ans"."id",
    "ans"."question_id",
    "ans"."answer" AS "Answer"
FROM "areas" "a", "dimensions" "d" , "questions" "q", "answers" "ans"
WHERE "a"."id" = "d"."area_id" AND "d"."id" = "q"."dimension_id" AND "q"."id" = "ans"."question_id" AND "ans"."answer" !=   '' AND "ans"."question_id" > 1590 AND "ans"."question_id" < 1619  ;

SELECT * FROM "questions" WHERE id = 1606;

SELECT * FROM "study_template_questions" WHERE "question_id" = 1606;

SELECT * FROM "study_template_questions" WHERE "study_template_id" = 135;


SELECT DISTINCT 
    "a"."name" AS "Area_Name", 
    "d"."name" AS "Dimension_Name",
    "ans"."answer" AS "Answer"
FROM "areas" "a", "dimensions" "d" , "questions" "q", "answers" "ans"
WHERE "a"."id" = "d"."area_id" AND "d"."id" = "q"."dimension_id" AND "q"."id" = "ans"."question_id" AND "ans"."answer" !=   '' AND ("ans"."question_id" < 1590 OR "ans"."question_id" > 1619  );
