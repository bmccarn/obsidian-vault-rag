{{/* Expand the chart name. */}}
{{- define "vault-rag.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Create a DNS-safe full release name. */}}
{{- define "vault-rag.fullname" -}}
{{- if contains (include "vault-rag.name" .) .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "vault-rag.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end }}

{{/* Labels that identify this chart version. */}}
{{- define "vault-rag.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | quote }}
{{ include "vault-rag.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/* Immutable workload and Service selector labels. */}}
{{- define "vault-rag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vault-rag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/* Prefer immutable image digests over mutable tags. */}}
{{- define "vault-rag.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- end -}}
{{- end }}

{{/* Labels that bind a workload to one runtime role. */}}
{{- define "vault-rag.componentSelectorLabels" -}}
{{ include "vault-rag.selectorLabels" .root }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/* Select the HTTP-serving pod role for the active backend. */}}
{{- define "vault-rag.serviceComponent" -}}
{{- if eq .Values.storage.backend "postgresql" -}}api{{- else -}}serve{{- end -}}
{{- end }}

{{/* Emit application storage config without chart-only PVC fields. */}}
{{- define "vault-rag.serviceStorage" -}}
{{- if eq .Values.storage.backend "postgresql" -}}
{{- $storage := dict
  "backend" "postgresql"
  "databaseUrlEnv" .Values.storage.databaseUrlEnv
  "pool" .Values.storage.pool
  "connectTimeout" .Values.storage.connectTimeout
  "statementTimeout" .Values.storage.statementTimeout
  "lockTimeout" .Values.storage.lockTimeout
  "idleTransactionTimeout" .Values.storage.idleTransactionTimeout
  "allowInsecureTransport" .Values.storage.allowInsecureTransport
  "cleanup" .Values.storage.cleanup
-}}
{{- with .Values.storage.ownerRole }}{{- $_ := set $storage "ownerRole" . }}{{- end -}}
{{- toYaml $storage -}}
{{- else -}}
backend: sqlite
{{- end -}}
{{- end }}

{{/* Validate role-specific safety invariants that JSON Schema cannot compare. */}}
{{- define "vault-rag.assertValues" -}}
{{- if eq .Values.storage.backend "postgresql" -}}
  {{- if eq .Values.api.databaseSecret.name .Values.worker.databaseSecret.name -}}
    {{- fail "api.databaseSecret and worker.databaseSecret must use distinct Secret names" -}}
  {{- end -}}
  {{- if and .Values.migrations.enabled (eq .Values.api.databaseSecret.name .Values.migrations.databaseSecret.name) -}}
    {{- fail "api.databaseSecret and migrations.databaseSecret must use distinct Secret names" -}}
  {{- end -}}
  {{- if and .Values.migrations.enabled (eq .Values.worker.databaseSecret.name .Values.migrations.databaseSecret.name) -}}
    {{- fail "worker.databaseSecret and migrations.databaseSecret must use distinct Secret names" -}}
  {{- end -}}
  {{- if eq .Values.api.runtimeSecret.name .Values.worker.runtimeSecret.name -}}
    {{- fail "api.runtimeSecret and worker.runtimeSecret must use distinct Secret names" -}}
  {{- end -}}
  {{- if gt .Values.storage.pool.minSize .Values.storage.pool.maxSize -}}
    {{- fail "storage.pool.minSize must not exceed storage.pool.maxSize" -}}
  {{- end -}}
  {{- if gt .Values.api.autoscaling.minReplicas .Values.api.autoscaling.maxReplicas -}}
    {{- fail "api.autoscaling.minReplicas must not exceed api.autoscaling.maxReplicas" -}}
  {{- end -}}
  {{- if gt .Values.api.pdb.minAvailable .Values.api.replicas -}}
    {{- fail "api.pdb.minAvailable must not exceed api.replicas" -}}
  {{- end -}}
  {{- if gt .Values.worker.pdb.minAvailable .Values.worker.replicas -}}
    {{- fail "worker.pdb.minAvailable must not exceed worker.replicas" -}}
  {{- end -}}
{{- else if ne (int .Values.storage.replicas) 1 -}}
  {{- fail "storage.replicas must be exactly one for SQLite" -}}
{{- end -}}
{{- range $name, $repository := .Values.serviceConfig.repositories -}}
  {{- if not (regexMatch "^https://[A-Za-z0-9.-]+(:[0-9]{1,5})?(/.*)?$" $repository.url) -}}
    {{- fail (printf "serviceConfig.repositories.%s.url must not contain userinfo" $name) -}}
  {{- end -}}
{{- end -}}
{{- end }}
