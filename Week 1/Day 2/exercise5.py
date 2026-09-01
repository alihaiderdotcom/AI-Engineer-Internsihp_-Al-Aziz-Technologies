# Exercise 5: Nested Data Structures

students = [
    {"name": "Ali", "age": 22, "courses": ["Python", "ML", "AI"]},
    {"name": "Sara", "age": 21, "courses": ["Python", "Web Dev"]},
    {"name": "Ahmed", "age": 23, "courses": ["ML", "AI"]}
]

# Print student info
print("--- Students Information ---")
for i, student in enumerate(students, 1):
    courses_str = ", ".join(student["courses"])
    print(f"Student {i}: {student['name']}, Age: {student['age']}, Courses: {courses_str}")

# Find students taking Python
print("\n--- Course Analysis ---")
python_students = [s["name"] for s in students if "Python" in s["courses"]]
print(f"Students taking Python: {', '.join(python_students)}")

# Create course enrollment dictionary
courses_dict = {}
for student in students:
    for course in student["courses"]:
        if course not in courses_dict:
            courses_dict[course] = []
        courses_dict[course].append(student["name"])

print(f"\nCourse enrollment:")
for course, enrolled_students in courses_dict.items():
    print(f"  {course}: {', '.join(enrolled_students)}")

# Count students in each course
print(f"\nCourse enrollment count:")
for course, enrolled_students in courses_dict.items():
    print(f"  {course}: {len(enrolled_students)} students")
