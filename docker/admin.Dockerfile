FROM node:20-slim

WORKDIR /app

COPY admin-dashboard/package.json admin-dashboard/package-lock.json ./
RUN npm ci

COPY admin-dashboard/ .

RUN npm run build

# Serve with a lightweight static server
RUN npm install -g serve
CMD ["serve", "-s", "dist", "-l", "3000"]
