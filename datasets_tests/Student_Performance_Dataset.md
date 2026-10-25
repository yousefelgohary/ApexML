# Student Performance Dataset

## About the Dataset

This dataset contains information about students' study habits, previous
academic performance, extracurricular activities, sleep, and practice
with sample question papers.

The main goal is to understand how these factors are related to
students' **Performance Index**.

## Columns

  ------------------------------------------------------------------------
  Column                               Description
  ------------------------------------ -----------------------------------
  `Hours Studied`                      Number of hours the student
                                       studied.

  `Previous Scores`                    Student's previous academic score.

  `Extracurricular Activities`         Whether the student participates in
                                       extracurricular activities (`Yes`
                                       or `No`).

  `Sleep Hours`                        Number of hours the student sleeps
                                       per day.

  `Sample Question Papers Practiced`   Number of sample question papers
                                       the student practiced.

  `Performance Index`                  The student's overall performance
                                       score. This is the value we want to
                                       predict.
  ------------------------------------------------------------------------

## Understanding the Data

The dataset provides information about several factors that may affect
student performance:

-   **Study time** --- how many hours the student studies.
-   **Previous performance** --- the student's previous score.
-   **Extracurricular activities** --- whether the student participates
    in extracurricular activities.
-   **Sleep** --- the student's daily sleep duration.
-   **Practice** --- how many sample question papers the student
    practiced.

## Target Column

`Performance Index`

This is the column that should be predicted using the other available
features.

## Example Question

A model can be used to estimate a student's **Performance Index** based
on their study hours, previous scores, extracurricular activities, sleep
hours, and number of practiced sample papers.
