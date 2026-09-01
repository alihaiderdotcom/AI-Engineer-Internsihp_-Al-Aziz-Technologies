# Exercise 3: Grade Calculator with Conditionals & Loops

marks = []
for i in range(5):
    mark = int(input(f"Subject {i+1} marks: "))
    marks.append(mark)

average = sum(marks) / len(marks)
print(f"\n--- Results ---")
print(f"All marks: {marks}")
print(f"Average: {average:.1f}")

# Determine grade
if average >= 90:
    grade = "A"
    feedback = "Excellent!"
elif average >= 80:
    grade = "B"
    feedback = "Good job!"
elif average >= 70:
    grade = "C"
    feedback = "Satisfactory"
elif average >= 60:
    grade = "D"
    feedback = "Passing, but needs improvement"
else:
    grade = "F"
    feedback = "Failed - requires retaking"

print(f"Grade: {grade}")
print(f"Feedback: {feedback}")

# Additional analysis
highest = max(marks)
lowest = min(marks)
print(f"\nHighest mark: {highest}")
print(f"Lowest mark: {lowest}")
