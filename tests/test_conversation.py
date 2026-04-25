from memory.conversation import ConversationState


def make_state(*pairs):
    """Build a ConversationState from (role, content) pairs."""
    s = ConversationState()
    for role, content in pairs:
        s.add(role, content)
    return s


# ------------------------------------------------------------------
# add()
# ------------------------------------------------------------------

def test_add_user_appends_message():
    s = ConversationState()
    s.add("user", "hello")
    assert s.messages == [{"role": "user", "content": "hello"}]


def test_add_user_increments_turn_count():
    s = ConversationState()
    s.add("user", "hi")
    s.add("user", "again")
    assert s.turn_count == 2


def test_add_assistant_does_not_increment_turn_count():
    s = ConversationState()
    s.add("assistant", "response")
    assert s.turn_count == 0


def test_add_preserves_order():
    s = make_state(("user", "a"), ("assistant", "b"), ("user", "c"))
    assert [m["role"] for m in s.messages] == ["user", "assistant", "user"]


# ------------------------------------------------------------------
# pop_oldest()
# ------------------------------------------------------------------

def test_pop_oldest_returns_removed_messages():
    s = make_state(("user", "first"), ("assistant", "second"), ("user", "third"))
    removed = s.pop_oldest(2)
    assert len(removed) == 2
    assert removed[0]["content"] == "first"
    assert removed[1]["content"] == "second"


def test_pop_oldest_updates_messages_in_place():
    s = make_state(("user", "first"), ("assistant", "second"), ("user", "third"))
    s.pop_oldest(2)
    assert len(s.messages) == 1
    assert s.messages[0]["content"] == "third"


def test_pop_oldest_zero_removes_nothing():
    s = make_state(("user", "a"), ("assistant", "b"))
    removed = s.pop_oldest(0)
    assert removed == []
    assert len(s.messages) == 2


def test_pop_oldest_all_leaves_empty():
    s = make_state(("user", "a"), ("assistant", "b"))
    s.pop_oldest(2)
    assert s.messages == []


# ------------------------------------------------------------------
# set_summary()
# ------------------------------------------------------------------

def test_set_summary_stores_text():
    s = ConversationState()
    s.set_summary("The user introduced themselves as Alice.")
    assert s.summary == "The user introduced themselves as Alice."


def test_set_summary_overwrites():
    s = ConversationState()
    s.set_summary("old")
    s.set_summary("new")
    assert s.summary == "new"


# ------------------------------------------------------------------
# clear()
# ------------------------------------------------------------------

def test_clear_empties_messages():
    s = make_state(("user", "a"), ("assistant", "b"))
    s.clear()
    assert s.messages == []


def test_clear_resets_turn_count():
    s = make_state(("user", "a"), ("user", "b"))
    s.clear()
    assert s.turn_count == 0


def test_clear_removes_summary():
    s = ConversationState()
    s.set_summary("some summary")
    s.clear()
    assert s.summary is None
