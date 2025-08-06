from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Arham"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=4, default=3, description= "A decimal value representing cgpa of student")

new_student = {"age": 32, "email": "abc@gmai.com"}

student = Student(**new_student)

print(student.model_dump_json())