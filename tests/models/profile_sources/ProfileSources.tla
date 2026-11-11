--------------------------- MODULE ProfileSources ---------------------------
EXTENDS Naturals, FiniteSets
CONSTANT Mode
Episodes == {1, 2}
VARIABLES written, live, members, indexed, removed, completed, queued, stage,
          snapshot, completedAtStart, chosen
vars == <<written, live, members, indexed, removed, completed, queued, stage,
          snapshot, completedAtStart, chosen>>
Init == /\ written = {} /\ live = {} /\ members = {} /\ indexed = {}
        /\ removed = {} /\ completed = {} /\ queued = FALSE /\ stage = "waiting"
        /\ snapshot = {} /\ completedAtStart = {} /\ chosen = {}
\* User pipeline persists; trigger_profile_clustering checks liveness then links.
Persist(e) == /\ e \notin written
              /\ written' = written \cup {e} /\ live' = live \cup {e}
              /\ UNCHANGED <<members,indexed,removed,completed,queued,stage,
                              snapshot,completedAtStart,chosen>>
Link(e) == /\ e \in live /\ e \notin members
           /\ members' = members \cup {e}
           /\ UNCHANGED <<written,live,indexed,removed,completed,queued,stage,
                           snapshot,completedAtStart,chosen>>
Emit == /\ 2 \in members /\ ~queued /\ queued' = TRUE
        /\ UNCHANGED <<written,live,members,indexed,removed,completed,stage,
                        snapshot,completedAtStart,chosen>>
\* Independent cascade; stale rows may remain indefinitely after deletion.
Index(e) == /\ e \in live /\ e \notin indexed
            /\ indexed' = indexed \cup {e}
            /\ UNCHANGED <<written,live,members,removed,completed,queued,stage,
                            snapshot,completedAtStart,chosen>>
\* Deleter source removal, member cleanup and successful return are separate.
\* Stuttering between these actions includes failed calls and retry delays.
Remove(e) == /\ e \in live
             /\ live' = live \ {e} /\ removed' = removed \cup {e}
             /\ UNCHANGED <<written,members,indexed,completed,queued,stage,
                             snapshot,completedAtStart,chosen>>
Detach(e) == /\ e \in removed /\ e \in members
             /\ members' = members \ {e}
             /\ UNCHANGED <<written,live,indexed,removed,completed,queued,stage,
                             snapshot,completedAtStart,chosen>>
Return(e) == /\ e \in removed /\ e \notin completed
             /\ (e \notin members \/ Mode = "skip_cleanup")
             /\ completed' = completed \cup {e}
             /\ UNCHANGED <<written,live,members,indexed,removed,queued,stage,
                             snapshot,completedAtStart,chosen>>
Erase(e) == /\ e \in removed /\ e \in indexed
            /\ indexed' = indexed \ {e}
            /\ UNCHANGED <<written,live,members,removed,completed,queued,stage,
                            snapshot,completedAtStart,chosen>>
\* extract_user_profile lists current scoped clusters after its profile lock.
Snapshot == /\ queued /\ stage = "waiting"
            /\ snapshot' = members /\ completedAtStart' = completed
            /\ stage' = "reading"
            /\ UNCHANGED <<written,live,members,indexed,removed,completed,queued,chosen>>
Select == /\ stage = "reading"
          /\ chosen' = (snapshot \cap indexed) \cup
              (CASE Mode = "index" -> {}
                    [] Mode = "fallback" -> IF snapshot # {} THEN {2} ELSE {}
                    [] OTHER -> IF 2 \in snapshot THEN {2} ELSE {})
          /\ stage' = "done"
          /\ UNCHANGED <<written,live,members,indexed,removed,completed,queued,
                          snapshot,completedAtStart>>
Next == (\E e \in Episodes: Persist(e) \/ Link(e) \/ Index(e) \/
                            Remove(e) \/ Detach(e) \/ Return(e) \/ Erase(e))
        \/ Emit \/ Snapshot \/ Select
Spec == Init /\ [][Next]_vars /\ WF_vars(Snapshot) /\ WF_vars(Select)
TypeOK == /\ written \subseteq Episodes /\ live \subseteq written
          /\ members \subseteq written /\ indexed \subseteq written
          /\ removed \subseteq written /\ completed \subseteq removed
          /\ queued \in BOOLEAN /\ stage \in {"waiting","reading","done"}
          /\ snapshot \subseteq Episodes /\ completedAtStart \subseteq Episodes
          /\ chosen \subseteq Episodes
NoCompletedDeletedSource == stage = "done" => chosen \cap completedAtStart = {}
CurrentSourceIncluded == (stage = "done" /\ 2 \in snapshot) => 2 \in chosen
SelectionFinishes == queued ~> (stage = "done")
=============================================================================
