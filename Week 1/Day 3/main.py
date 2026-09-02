"""Entry point for the Week 1 Day 3 OOP exercise."""

from calculator import calculate_average, get_grade, summarize_scores
from student import Intern


student = Intern("Ali Haider", [85, 90, 78, 88, 92])
average = student.average_score(calculate_average)
grade = get_grade(average)
subjects = student.subjects("Python", "Machine Learning", "AI")
summary = summarize_scores(*student.scores, learner=student.name)

print(f"Student: {student.name}")
print(f"Average: {average:.2f}")
print(f"Grade: {grade}")
print(f"Role: {student.role}")
print(f"Subjects: {', '.join(subjects)}")
print(student.introduce())
print(f"Score count: {summary['count']}")
