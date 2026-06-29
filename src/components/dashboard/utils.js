export const DEFAULT_COLOR_TAG = "#4F46E5";
export const COLOR_SWATCHES = ["#4F46E5", "#10B981", "#06B6D4", "#F59E0B", "#F43F5E"];

export const classNames = (...classes) => classes.filter(Boolean).join(" ");

export const formatHours = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) {
    return "0.00";
  }

  return numeric.toFixed(numeric >= 10 ? 1 : 2);
};

export const formatUnixDate = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return "Never";
  }

  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
  }).format(new Date(numeric * 1000));
};

export const formatIsoDate = (value) => {
  if (!value) {
    return "Not configured";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Invalid date";
  }

  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
};

export const getExpiryState = (expiresAt) => {
  if (!expiresAt) {
    return {
      tone: "neutral",
      label: "No expiration configured",
      description: "Add lifecycle metadata when this account has a known renewal window.",
    };
  }

  const expires = new Date(expiresAt);
  if (Number.isNaN(expires.getTime())) {
    return {
      tone: "danger",
      label: "Invalid expiration metadata",
      description: "The stored expiration timestamp cannot be parsed.",
    };
  }

  const remainingMs = expires.getTime() - Date.now();
  if (remainingMs <= 0) {
    return {
      tone: "danger",
      label: "Expired",
      description: "This managed account is past its configured expiration date.",
    };
  }

  const remainingHours = Math.ceil(remainingMs / 3600000);
  const remainingDays = Math.floor(remainingHours / 24);
  const label =
    remainingDays >= 1
      ? `${remainingDays} day${remainingDays === 1 ? "" : "s"} remaining`
      : `${remainingHours} hour${remainingHours === 1 ? "" : "s"} remaining`;

  return {
    tone: remainingHours <= 72 ? "warning" : "success",
    label,
    description: `Expires ${formatIsoDate(expiresAt)}.`,
  };
};

export const getInitials = (profile) => {
  const source = profile?.name || profile?.email || "Aion";
  return source
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part.charAt(0).toUpperCase())
    .join("");
};

export const profileNameError = (name, profiles, currentProfileId = null) => {
  const normalized = name.trim();
  if (!normalized) {
    return "Profile name is required.";
  }
  if (normalized.length > 80) {
    return "Profile name cannot exceed 80 characters.";
  }

  const duplicate = profiles.some(
    (profile) =>
      profile.id !== currentProfileId &&
      profile.name.trim().toLowerCase() === normalized.toLowerCase(),
  );
  if (duplicate) {
    return "A profile with this name already exists.";
  }

  return "";
};

export const validateEmail = (value) => {
  const email = value.trim();
  if (!email) {
    return "Account email is required.";
  }
  if (email.length > 254) {
    return "Account email cannot exceed 254 characters.";
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return "Enter a valid account email.";
  }
  return "";
};

export const validateProxyHostPort = (value, proxyHasAnyValue) => {
  const hostPort = value.trim();
  if (!hostPort) {
    return proxyHasAnyValue ? "Proxy host:port is required when proxy credentials are set." : "";
  }
  if (hostPort.length > 255) {
    return "Proxy host:port cannot exceed 255 characters.";
  }
  if (/\s|@|\/|\\|\.\.|:\/\//.test(hostPort)) {
    return "Use plain host:port without schemes, credentials, paths, or traversal segments.";
  }

  const match = hostPort.match(/^([A-Za-z0-9._\-[\]]+):(\d{1,5})$/);
  if (!match) {
    return "Proxy must be formatted as host:port.";
  }

  const port = Number(match[2]);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    return "Proxy port must be between 1 and 65535.";
  }

  return "";
};
