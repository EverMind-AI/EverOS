interface ExampleQueriesProps {
  onSelectQuery: (query: string) => void;
  disabled: boolean;
}

const EXAMPLE_QUERIES = [
  'What body parts did Gared lose to the cold?',
  'How many years had Will been on the Wall before the prologue events?',
  'What was Ser Waymar Royce\'s cloak made of?',
  'Describe the Other\'s sword that fought Ser Waymar.',
];

export function ExampleQueries({ onSelectQuery, disabled }: ExampleQueriesProps) {
  return (
    <div className="example-queries">
      <h3>Example Questions:</h3>
      <div className="example-queries-list">
        {EXAMPLE_QUERIES.map((query, index) => (
          <button
            key={index}
            className="example-query-button"
            onClick={() => onSelectQuery(query)}
            disabled={disabled}
          >
            {query}
          </button>
        ))}
      </div>
    </div>
  );
}
