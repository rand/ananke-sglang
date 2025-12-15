{{/*
Expand the name of the chart.
*/}}
{{- define "sglang-ananke.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "sglang-ananke.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "sglang-ananke.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "sglang-ananke.labels" -}}
helm.sh/chart: {{ include "sglang-ananke.chart" . }}
{{ include "sglang-ananke.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "sglang-ananke.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sglang-ananke.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "sglang-ananke.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "sglang-ananke.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Build server command arguments
*/}}
{{- define "sglang-ananke.serverArgs" -}}
- --model-path
- {{ .Values.model.path | quote }}
- --host
- "0.0.0.0"
- --port
- {{ .Values.server.port | quote }}
- --tp-size
- {{ .Values.server.tpSize | quote }}
- --mem-fraction-static
- {{ .Values.server.memFraction | quote }}
{{- if .Values.ananke.enabled }}
- --grammar-backend
- ananke
- --ananke-language
- {{ .Values.ananke.language | quote }}
- --ananke-max-rollback-tokens
- {{ .Values.ananke.maxRollbackTokens | quote }}
{{- if .Values.ananke.enabledDomains }}
- --ananke-enabled-domains
- {{ .Values.ananke.enabledDomains | quote }}
{{- end }}
{{- end }}
{{- range .Values.server.extraArgs }}
- {{ . | quote }}
{{- end }}
{{- end }}

{{/*
Generate environment variables
*/}}
{{- define "sglang-ananke.env" -}}
{{- if .Values.model.existingSecret }}
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ .Values.model.existingSecret.name }}
      key: {{ .Values.model.existingSecret.key }}
{{- else if .Values.model.hfToken }}
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ include "sglang-ananke.fullname" . }}-hf-token
      key: token
{{- end }}
- name: SGLANG_LOG_LEVEL
  value: {{ .Values.server.logLevel | quote }}
- name: SGLANG_ENABLE_METRICS
  value: {{ .Values.server.enableMetrics | quote }}
{{- if .Values.ananke.defaultConstraintSpec }}
- name: ANANKE_DEFAULT_CONSTRAINT_SPEC
  value: {{ .Values.ananke.defaultConstraintSpec | quote }}
{{- end }}
{{- with .Values.extraEnv }}
{{- toYaml . }}
{{- end }}
{{- end }}

{{/*
Generate volume mounts
*/}}
{{- define "sglang-ananke.volumeMounts" -}}
- name: shm
  mountPath: /dev/shm
{{- if .Values.modelVolume.enabled }}
- name: model-cache
  mountPath: {{ .Values.modelVolume.mountPath }}
{{- end }}
- name: localtime
  mountPath: /etc/localtime
  readOnly: true
{{- with .Values.extraVolumeMounts }}
{{- toYaml . }}
{{- end }}
{{- end }}

{{/*
Generate volumes
*/}}
{{- define "sglang-ananke.volumes" -}}
- name: shm
  emptyDir:
    medium: Memory
    sizeLimit: 10Gi
{{- if .Values.modelVolume.enabled }}
- name: model-cache
  {{- if eq .Values.modelVolume.type "pvc" }}
  persistentVolumeClaim:
    claimName: {{ .Values.modelVolume.pvc.existingClaim | default (printf "%s-model-cache" (include "sglang-ananke.fullname" .)) }}
  {{- else if eq .Values.modelVolume.type "hostPath" }}
  hostPath:
    path: {{ .Values.modelVolume.hostPath.path }}
    type: {{ .Values.modelVolume.hostPath.type }}
  {{- else }}
  emptyDir: {}
  {{- end }}
{{- end }}
- name: localtime
  hostPath:
    path: /etc/localtime
    type: File
{{- with .Values.extraVolumes }}
{{- toYaml . }}
{{- end }}
{{- end }}
