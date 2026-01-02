.PHONY: build up down logs shell

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f jupyter

shell:
	docker exec -it housing-jupyter bash
