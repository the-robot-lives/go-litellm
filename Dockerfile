# go-litellm stage gateway image (GNP search-v2, ADR-021).
#
# Build (multiarch — the EKS fleet is mixed arm64/amd64) and push straight
# to ECR, versioned tag only (never :latest):
#
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     --push -t 783147025407.dkr.ecr.us-east-1.amazonaws.com/gnp-litellm:v1.0.0-eks.1 .
#
# Deployed by helm/gnp-litellm (stage-only chart). The ENTRYPOINT flag
# surface matches what the chart expects: listen on all interfaces, port
# 4445, config mounted at /etc/go-litellm/config.yaml.
FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath \
    -ldflags '-s -w -X github.com/noizu-labs/go-litellm/internal/version.Version=v1.0.0-eks.1' \
    -o /out/go-litellm ./cmd/go-litellm

# distroless/static: CA certs included (needed for the DashScope HTTPS call),
# no shell, nonroot by default.
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/go-litellm /go-litellm
# Request-log sqlite lives on the chart's emptyDir at /var/lib/go-litellm.
WORKDIR /var/lib/go-litellm
ENTRYPOINT ["/go-litellm", "--host", "0.0.0.0", "--port", "4445", "--config", "/etc/go-litellm/config.yaml"]
