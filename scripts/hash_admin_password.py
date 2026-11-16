"""Generate an ADMIN_PASSWORD_HASH without placing plaintext in shell history."""

from getpass import getpass

from visionops.api.auth import hash_password


def main() -> None:
    password = getpass("Admin password: ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match")
    print(hash_password(password))


if __name__ == "__main__":
    main()
