import { redirect } from "next/navigation";

// Only `phrase-vae` exists today; redirect to its listing.  When
// other variants land, this becomes a real section index.
export default function ModelsIndex() {
  redirect("/models/phrase-vae");
}
