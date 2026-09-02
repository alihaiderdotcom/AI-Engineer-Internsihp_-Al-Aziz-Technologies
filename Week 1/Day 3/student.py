"""Student classes used by the Day 3 application."""


class Student:
    """Represent a student and their subject scores."""

    def __init__(self, name, scores):
        if not scores:
            raise ValueError("scores must contain at least one value")
        self.name = name
        self.scores = list(scores)

    def average_score(self, calculator):
        """Calculate this student's average using an injected function."""
        return calculator(self.scores)

    def subjects(self, *subject_names):
        """Return the supplied subjects as a list."""
        return list(subject_names)


class Intern(Student):
    """A Student with an internship role."""

    role = "AI Engineering Intern"

    def introduce(self):
        return f"I am {self.name}, an {self.role}."
