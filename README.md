# smart-rafic-light-controller
college 8th sem final project

# Project setup
1. to start the roject install postgresql for database
2. install redis 
3. install python > 3.10
4. create virtual environment
    - python -m venv venv
5. Activate the virtual environment by running the command:
        - source venv/bin/activate (Linux or mac)
        OR - .\venv\Scripts\activate (windows)
6. Install the required dependencies by running the command:
        - pip install -r requirements.txt
7. Next, apply the database migrations by running:
        - python manage.py migrate
8. Create a superuser account by running:
        - python manage.py createsuperuser
9. to run project locally
     -  run python manage.py runserver 
10. Start the Celery worker by running:
     -  celery -A config worker --pool=solo -l info (Windows)
     -  celery -A config worker -l info (linux or mac)
11. Start the Celery beat by running:
        - celery -A config beat -l info

# Note eun python, celery and celery-beat commands in seprate terminals
