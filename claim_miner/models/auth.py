from __future__ import annotations

from typing import List, Union

from sqlalchemy import BigInteger, ForeignKey, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import ARRAY, ENUM
from sqlalchemy.orm import Mapped, mapped_column

from ..pyd_models import UserModel, topic_type, permission
from .base import Topic

permission_db = ENUM(permission, name="permission")


class User(Topic):
    """ClaimMiner users."""

    __tablename__ = "user"
    pyd_model = UserModel

    __mapper_args__ = {
        "polymorphic_load": "inline",
        "polymorphic_identity": topic_type.agent,
    }

    id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Topic.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )  #: Primary key
    handle: Mapped[String] = mapped_column(String)  #: username
    passwd: Mapped[String] = mapped_column(String)  #: password (scram hash)
    email: Mapped[String] = mapped_column(String)  #: email
    name: Mapped[String] = mapped_column(String)  #: name
    confirmed: Mapped[Boolean] = mapped_column(
        Boolean, server_default="false"
    )  #: account confirmed by admin
    created: Mapped[DateTime] = mapped_column(
        DateTime, server_default="now()"
    )  #: date of creation
    external_id: Mapped[String] = mapped_column(
        String, unique=True
    )  #: external id (e.g. google id)
    picture_url: Mapped[String] = mapped_column(String)  #: picture url
    permissions: Mapped[List[permission]] = mapped_column(
        ARRAY(permission_db)
    )  #: User's global permissions

    def can(self, perm: Union[str, permission]):
        "Does the user have this permission?"
        permissions = self.permissions or []
        perm = permission[perm] if isinstance(perm, str) else perm
        return (perm in permissions) or (permission.admin in permissions)
